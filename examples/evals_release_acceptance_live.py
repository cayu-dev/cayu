"""Credential-gated release acceptance for the zero-code Evals journey.

OpenAI:
    CAYU_PROVIDER=openai uv run python examples/evals_release_acceptance_live.py

Anthropic:
    CAYU_PROVIDER=anthropic uv run python examples/evals_release_acceptance_live.py

The check scaffolds an ordinary project, starts it through ``cayu serve --dev``,
and performs three candidate executions: one Control Plane session, one
production-first fresh trial, and one author-first judged trial. It proves that
an explicit TOML judge declaration becomes bounded tool-free authority, then
round-trips captured, fresh, and judged artifacts through public CLI reports
and stable comparison exits without constructing Evals-specific Python
configuration. Provider-level retry behavior is left to the selected provider
implementation.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.parse import parse_qs, urlsplit
from urllib.request import urlopen

from playwright.async_api import Page, async_playwright, expect

import cayu
from _live_checks import is_superseded_browser_read_abort, require, require_equal

AGENT_NAME = "eval-release-acceptance"
EVIDENCE_PREFIX = "CAYU_NIGHTLY_EVIDENCE="
EXPECTED_MARKER = "CAYU_EVAL_RELEASE_ACCEPTANCE_OK"
PROJECT_NAME = "eval_release_acceptance"


async def main() -> None:
    provider_name = _provider_name()
    _require_api_key(provider_name)
    model_name = _model(provider_name)

    with tempfile.TemporaryDirectory(prefix="cayu-evals-release-live-") as temporary:
        temporary_root = Path(temporary)
        project_root = _scaffold_project(temporary_root, provider_name, model_name)
        port = _unused_loopback_port()
        base_url = f"http://127.0.0.1:{port}"
        log_path = temporary_root / "serve.log"
        environment = os.environ.copy()
        environment.update(
            {
                "CAYU_MODEL": model_name,
                "CAYU_PROVIDER": provider_name,
                "CAYU_RELEASE_ID": f"{provider_name}-generated-project-release",
                "PYTHONPATH": _generated_server_pythonpath(project_root),
                "PYTHONUNBUFFERED": "1",
            }
        )

        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-P",
                    "-m",
                    "cayu",
                    "serve",
                    "--dev",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                ],
                cwd=project_root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                await asyncio.to_thread(_wait_for_server, process, base_url, log_path)
                evidence = await _run_browser_journey(base_url, project_root)
            except Exception as exc:
                await asyncio.to_thread(_stop_process, process)
                raise RuntimeError(
                    "generated-project Evals acceptance failed; server tail follows:\n"
                    f"{_read_log(log_path)}"
                ) from exc
            finally:
                await asyncio.to_thread(_stop_process, process)

        evidence.update(
            {
                "provider": provider_name,
                "model": model_name,
                "generated_project": True,
                "source_session_completed": True,
            }
        )
        print(f"{EVIDENCE_PREFIX}{json.dumps(evidence, sort_keys=True)}")


def _scaffold_project(root: Path, provider_name: str, model_name: str) -> Path:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "cayu",
            "new",
            PROJECT_NAME,
            "--agent-name",
            AGENT_NAME,
            "--provider",
            provider_name,
            "--dir",
            str(root),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"cayu new exited {completed.returncode}: {completed.stdout}{completed.stderr}"
        )
    project_root = root / PROJECT_NAME
    require(project_root.is_dir(), "cayu new must create the requested project")
    pyproject = project_root / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8")
        + '\n[tool.cayu.evals]\nprice_book = "bundled-public"\n'
        + "\n[tool.cayu.evals.default_judge]\n"
        + f"provider = {json.dumps(provider_name, ensure_ascii=False)}\n"
        + f"model = {json.dumps(model_name, ensure_ascii=False)}\n"
        + 'privacy_policy = "public-only"\n'
        + "allow_same_model = true\n"
        + "timeout_seconds = 120\n"
        + "max_input_tokens = 4096\n"
        + "max_output_tokens = 1024\n"
        + "max_total_tokens = 5120\n"
        + 'max_estimated_cost = "0.1"\n'
        + 'cost_currency = "USD"\n',
        encoding="utf-8",
    )
    return project_root


def _generated_server_pythonpath(project_root: Path) -> str:
    """Load the generated project with the exact Cayu package under acceptance."""

    package_file = cayu.__file__
    if package_file is None:
        raise RuntimeError("the loaded Cayu package has no filesystem import root")
    cayu_import_root = Path(package_file).resolve().parent.parent
    return os.pathsep.join((str(cayu_import_root), str(project_root.resolve())))


async def _run_browser_journey(base_url: str, project_root: Path) -> dict[str, object]:
    failures: dict[str, list[str]] = {
        "console": [],
        "page": [],
        "request": [],
        "api": [],
    }
    deferred_source_run_aborts: list[str] = []
    journey_state = {"source_session_terminal": False}
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1440, "height": 1000},
            accept_downloads=True,
        )
        page = await context.new_page()
        _record_browser_failures(
            page,
            failures,
            deferred_source_run_aborts=deferred_source_run_aborts,
        )
        try:
            await _require_declared_judge_profile(page, base_url)
            await page.goto(f"{base_url}/cayu/run", wait_until="networkidle")
            await expect(page.get_by_role("heading", name="New Run", exact=True)).to_be_visible()
            await page.get_by_label("Agent", exact=True).select_option(AGENT_NAME)
            await page.get_by_label("Prompt", exact=True).fill(
                f"Reply with exactly {EXPECTED_MARKER} and no other text."
            )
            await page.get_by_role("button", name="Run", exact=True).click()
            await _require_source_session_completed(page)
            journey_state["source_session_terminal"] = True
            session_id = (await page.get_by_test_id("run-session-reference").inner_text()).strip()
            require(bool(session_id), "the Control Plane run must publish a session reference")
            await page.get_by_role("button", name="View Session →", exact=True).click()
            await expect(page).to_have_url(re.compile(r"/cayu/sessions/[^/?]+$"))

            await page.get_by_test_id("evaluate-session").click()
            sheet = page.get_by_test_id("promotion-sheet")
            await expect(sheet).to_be_visible()
            await sheet.get_by_label("Case name", exact=True).fill(
                "Real-provider release regression"
            )
            await sheet.get_by_label("Assertion quick-add type", exact=True).select_option(
                "final_output_contains"
            )
            await sheet.get_by_role("button", name="Add observed", exact=True).click()
            output_assertion = sheet.get_by_test_id("promotion-assertion").nth(1)
            await expect(output_assertion).to_have_count(1)
            await output_assertion.get_by_label("Assertion ID", exact=True).fill(
                "output-contains-marker"
            )
            await output_assertion.get_by_label("Type", exact=True).select_option(
                "final_output_contains"
            )
            await output_assertion.get_by_label("Expected output text", exact=True).fill(
                EXPECTED_MARKER
            )
            fresh_execution = sheet.get_by_test_id("promotion-fresh-execution")
            await expect(fresh_execution).to_have_count(1)
            await fresh_execution.get_by_label("Trial timeout (seconds)", exact=True).fill("120")
            await fresh_execution.get_by_label("Max model steps per trial", exact=True).fill("2")
            await fresh_execution.get_by_label("Max total tokens per trial", exact=True).fill(
                "4096"
            )
            await fresh_execution.get_by_label("Max tool calls per trial", exact=True).fill("1")
            await fresh_execution.get_by_label("Max run time per trial (seconds)", exact=True).fill(
                "120"
            )
            await fresh_execution.get_by_label("Max estimated cost per trial", exact=True).fill(
                "0.1"
            )
            await sheet.get_by_test_id("promotion-preview").click()
            await expect(
                sheet.get_by_text("This score matches the current edits.", exact=True)
            ).to_be_visible(timeout=20_000)

            async with page.expect_download() as corpus_download_info:
                await sheet.get_by_test_id("promotion-export").click()
            corpus_download = await corpus_download_info.value
            corpus_path_text = await corpus_download.path()
            require(corpus_path_text is not None, "evaluation export must download a corpus")
            corpus_path = Path(corpus_path_text)
            corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
            require(
                re.fullmatch(r"eval\.[0-9a-f]{64}", corpus["target_key"]) is not None,
                "the scaffold must publish a generated eval target",
            )
            require(
                session_id not in json.dumps(corpus, sort_keys=True),
                "the portable corpus must not expose the source session ID",
            )
            require_equal(
                [assertion["kind"] for assertion in corpus["cases"][0]["assertions"]],
                ["root_status", "final_output_contains"],
                "the exported case must retain both reviewed assertions",
            )

            await sheet.get_by_test_id("promotion-launch").click()
            await expect(
                sheet.get_by_text(re.compile(r"Started current-app eval run .+\."))
            ).to_be_visible(timeout=20_000)
            await sheet.get_by_role("button", name="Approve baseline", exact=True).click()
            await expect(sheet.get_by_text("Baseline approved", exact=True)).to_be_visible()
            await sheet.get_by_role("link", name="Open run", exact=True).click()

            run_ids = parse_qs(urlsplit(page.url).query).get("run", [])
            require_equal(len(run_ids), 1, "fresh launch must select one durable eval run")
            run_id = run_ids[0]
            await _require_completed_result(page)
            await expect(
                page.get_by_text("These results are comparable.", exact=True)
            ).to_be_visible()
            await expect(
                page.get_by_text("No compatible-result regressions.", exact=True)
            ).to_be_visible()

            fresh_path = await _download_fresh_result(page, run_id)
            await _download_fresh_html_report(page, run_id)
            fresh = json.loads(fresh_path.read_text(encoding="utf-8"))
            require_equal(
                fresh["result"]["run"]["status"],
                "passed",
                "the fresh generated-project eval must pass",
            )

            await page.get_by_role("tab", name="Results", exact=True).click()
            captured_row = page.get_by_role("row").filter(has_text="Captured")
            await expect(captured_row).to_have_count(1)
            await captured_row.get_by_role("button").click()
            captured_path = await _download_catalog_result(page, "json")
            await _download_catalog_result(page, "html")
            captured = json.loads(captured_path.read_text(encoding="utf-8"))
            require_equal(
                captured["presentation"]["origin"],
                "captured_session",
                "the baseline artifact must retain its captured origin",
            )
            require(
                session_id not in json.dumps(captured, sort_keys=True),
                "the captured result must not expose the source session ID",
            )

            judged_path = await _run_authored_judged_evaluation(page, base_url)

            cli_evidence = await _run_cli_gate(
                project_root=project_root,
                corpus_path=corpus_path,
                captured_path=captured_path,
                fresh_path=fresh_path,
                judged_path=judged_path,
            )
            _require_no_browser_failures(
                failures,
                deferred_source_run_aborts=deferred_source_run_aborts,
                source_session_terminal=journey_state["source_session_terminal"],
            )
            return {
                "candidate_executions": 3,
                "deterministic_assertions": 3,
                "judge_executions": 1,
                "browser": "chromium",
                "comparison_regressions": 0,
                "local_ci": cli_evidence,
            }
        finally:
            await browser.close()


async def _require_declared_judge_profile(page: Page, base_url: str) -> None:
    response = await page.request.get(f"{base_url}/api/evals/targets")
    require_equal(response.status, 200, "the generated eval target catalog must be readable")
    catalog = await response.json()
    require_equal(len(catalog["items"]), 1, "the generated project must publish one eval target")
    target = catalog["items"][0]
    require_equal(
        len(target["judge_profiles"]),
        1,
        "the explicit project judge declaration must publish one trusted profile",
    )
    require_equal(
        target["judge_profiles"][0]["same_model_use"],
        "allowed_and_labeled",
        "the generated same-model judge must retain explicit opt-in",
    )
    require_equal(
        target["judge_profile_routes"][0]["candidate_route_relation"],
        "same_model",
        "the catalog must label the actual candidate-to-judge route relation",
    )
    require_equal(
        target["cost_budget_currencies"],
        ["USD"],
        "the explicit project price book must publish one candidate budget currency",
    )
    require_equal(
        target["judge_profiles"][0]["max_estimated_cost"],
        "0.1",
        "the generated judge profile must publish its declared cost ceiling",
    )


async def _run_authored_judged_evaluation(page: Page, base_url: str) -> Path:
    await page.get_by_test_id("new-evaluation").click()
    authoring = page.get_by_test_id("eval-suite-authoring-sheet")
    await expect(authoring).to_be_visible()
    await authoring.get_by_test_id("authored-suite-id").fill("real-provider-quality")
    await authoring.get_by_test_id("authored-suite-name").fill("Real-provider quality")
    await authoring.get_by_test_id("authored-case-id").fill("exact-release-marker")
    await authoring.get_by_test_id("authored-case-name").fill("Exact release marker")
    await authoring.get_by_role("textbox", name="Case input message 1", exact=True).fill(
        f"Reply with exactly {EXPECTED_MARKER} and no other text."
    )
    await authoring.get_by_role("button", name="Add same-model AI judge", exact=True).click()
    judge = authoring.get_by_test_id("structured-judge-editor")
    await expect(judge).to_have_count(1)
    await expect(judge.get_by_test_id("judge-profile-summary")).to_contain_text(
        "Same model as candidate"
    )
    reference_mode = judge.locator('select[id$="-reference-mode"]')
    await expect(reference_mode).to_have_count(1)
    await reference_mode.select_option("public_reference")
    expected_answer = judge.locator('textarea[id$="-expected-answer"]')
    await expect(expected_answer).to_have_count(1)
    await expected_answer.fill(EXPECTED_MARKER)
    await authoring.get_by_label("Timeout per trial", exact=True).fill("120")
    await authoring.get_by_label("Maximum cost per trial", exact=True).fill("0.1")
    await authoring.get_by_test_id("authored-suite-preview").click()
    await expect(authoring.get_by_text("Suite is ready to save", exact=True)).to_be_visible(
        timeout=20_000
    )
    await authoring.get_by_test_id("authored-suite-save").click()
    await expect(authoring.get_by_text(re.compile(r"Saved immutable suite .+\."))).to_be_visible()
    await authoring.get_by_test_id("authored-suite-run-preview").click()
    await expect(authoring.get_by_text("1 durable run ready", exact=True)).to_be_visible()
    await expect(authoring.get_by_test_id("authored-suite-exposure")).to_contain_text(
        "candidate cost not hard bounded"
    )
    await expect(authoring.get_by_test_id("authored-suite-exposure")).to_contain_text("0.1 USD")
    await authoring.get_by_test_id("authored-suite-launch").click()
    await expect(
        authoring.get_by_text("Started 1 durable eval run for 1 selected case.", exact=True)
    ).to_be_visible()
    run_ids = parse_qs(urlsplit(page.url).query).get("run", [])
    require_equal(len(run_ids), 1, "the judged launch must select one durable eval run")
    run_id = run_ids[0]
    await page.keyboard.press("Escape")
    status = page.get_by_test_id("eval-run-status-announcement")
    await expect(status).to_have_text("Eval run status: completed.", timeout=330_000)
    await expect(page.get_by_text("same model · explicitly allowed", exact=True)).to_be_visible()
    await expect(page.get_by_text("correctness", exact=True)).to_be_visible()
    await expect(page.get_by_text("passed", exact=True).first).to_be_visible()

    response = await page.request.get(f"{base_url}/api/evals/runs/{run_id}/result")
    require_equal(response.status, 200, "the judged result must be readable")
    result = await response.json()
    assertions = result["result"]["run"]["cases"][0]["trials"][0]["assertions"]
    structured = next(
        item for item in assertions if item["detail"]["kind"] == "structured_model_judge"
    )
    require_equal(structured["outcome"], "passed", "the real structured judge must pass")
    require_equal(
        structured["detail"]["criteria"][0]["criterion_id"],
        "correctness",
        "the result must retain the exact rubric criterion",
    )

    async with page.expect_download() as download_info:
        await page.get_by_role("button", name="JSON", exact=True).click()
    download = await download_info.value
    require_equal(
        download.suggested_filename,
        f"{run_id}.eval-result.json",
        "the judged result must retain its server-owned filename",
    )
    path = await download.path()
    require(path is not None, "the judged result download must be readable")
    async with page.expect_download() as html_download_info:
        await page.get_by_role("button", name="HTML", exact=True).click()
    html_download = await html_download_info.value
    require_equal(
        html_download.suggested_filename,
        f"{run_id}.eval-report.html",
        "the judged HTML report must retain its server-owned filename",
    )
    return Path(path)


async def _require_completed_result(page: Page) -> None:
    status = page.get_by_test_id("eval-run-status-announcement")
    await expect(status).to_have_text(
        re.compile(r"^Eval run status: (completed|failed|cancelled)\.$"),
        timeout=330_000,
    )
    require_equal(
        await status.inner_text(),
        "Eval run status: completed.",
        "the fresh generated-project eval must complete",
    )
    published = page.locator('[data-slot="card-title"]').filter(has_text="Published result")
    await expect(published).to_be_visible(timeout=20_000)
    await expect(page.get_by_text("passed", exact=True).first).to_be_visible()
    root_assertion = page.get_by_role("row").filter(has_text="root-status")
    await expect(root_assertion).to_have_count(1)
    await expect(root_assertion.get_by_text("passed", exact=True)).to_be_visible()
    output_assertion = page.get_by_role("row").filter(has_text="output-contains-marker")
    await expect(output_assertion).to_have_count(1)
    await expect(output_assertion.get_by_text("passed", exact=True)).to_be_visible()


async def _require_source_session_completed(page: Page) -> None:
    terminal = page.locator('[data-mutation-transport-phase="terminal"]')
    await expect(terminal).to_be_visible(timeout=330_000)
    completed = page.get_by_text("Session completed", exact=True)
    if await completed.count() == 1 and await completed.is_visible():
        return
    failure = page.get_by_text(re.compile(r"^Session failed:"))
    detail = await failure.first.inner_text() if await failure.count() else "unknown terminal state"
    raise AssertionError(f"generated-project source session did not complete: {detail}")


async def _download_fresh_result(page: Page, run_id: str) -> Path:
    async with page.expect_download() as download_info:
        await page.get_by_role("button", name="JSON", exact=True).click()
    download = await download_info.value
    require_equal(
        download.suggested_filename,
        f"{run_id}.eval-result.json",
        "the fresh-result download must retain its server-owned filename",
    )
    path = await download.path()
    require(path is not None, "the fresh-result download must be readable")
    return Path(path)


async def _download_fresh_html_report(page: Page, run_id: str) -> None:
    async with page.expect_download() as download_info:
        await page.get_by_role("button", name="HTML", exact=True).click()
    download = await download_info.value
    require_equal(
        download.suggested_filename,
        f"{run_id}.eval-report.html",
        "the fresh HTML download must retain its server-owned filename",
    )


async def _download_catalog_result(page: Page, format_name: str) -> Path:
    button_name = "JSON" if format_name == "json" else "HTML"
    async with page.expect_download() as download_info:
        await page.get_by_role("button", name=button_name, exact=True).click()
    download = await download_info.value
    require(
        re.fullmatch(
            r"[0-9a-f]{64}\.eval-(?:result\.json|report\.html)",
            download.suggested_filename,
        )
        is not None,
        "the catalog report must use its immutable result revision as the filename",
    )
    path = await download.path()
    require(path is not None, "the catalog result download must be readable")
    return Path(path)


async def _run_cli_gate(
    *,
    project_root: Path,
    corpus_path: Path,
    captured_path: Path,
    fresh_path: Path,
    judged_path: Path,
) -> dict[str, object]:
    def run(arguments: list[str]) -> None:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = _generated_server_pythonpath(project_root)
        completed = subprocess.run(
            [sys.executable, "-P", "-m", "cayu", "eval", *arguments],
            cwd=project_root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if completed.returncode != 0:
            raise AssertionError(
                f"cayu eval {' '.join(arguments)} exited {completed.returncode}: "
                f"{completed.stdout}{completed.stderr}"
            )

    with tempfile.TemporaryDirectory(prefix="cayu-evals-release-cli-") as temporary:
        output_root = Path(temporary)
        captured_json = output_root / "captured-report.json"
        captured_html = output_root / "captured-report.html"
        fresh_html = output_root / "fresh-report.html"
        judged_json = output_root / "judged-report.json"
        judged_html = output_root / "judged-report.html"
        comparison_json = output_root / "comparison.json"
        comparison_html = output_root / "comparison.html"
        judged_comparison_json = output_root / "judged-comparison.json"

        await asyncio.to_thread(run, ["validate", str(corpus_path)])
        await asyncio.to_thread(
            run,
            ["report", str(captured_path), "--json", "--output", str(captured_json)],
        )
        await asyncio.to_thread(
            run,
            ["report", str(captured_path), "--html", "--output", str(captured_html)],
        )
        await asyncio.to_thread(
            run,
            ["report", str(fresh_path), "--html", "--output", str(fresh_html)],
        )
        await asyncio.to_thread(
            run,
            ["report", str(judged_path), "--json", "--output", str(judged_json)],
        )
        await asyncio.to_thread(
            run,
            ["report", str(judged_path), "--html", "--output", str(judged_html)],
        )
        await asyncio.to_thread(
            run,
            [
                "compare",
                str(captured_path),
                str(fresh_path),
                "--json",
                "--output",
                str(comparison_json),
            ],
        )
        await asyncio.to_thread(
            run,
            [
                "compare",
                str(captured_path),
                str(fresh_path),
                "--html",
                "--output",
                str(comparison_html),
            ],
        )
        await asyncio.to_thread(
            run,
            [
                "compare",
                str(judged_path),
                str(judged_path),
                "--json",
                "--output",
                str(judged_comparison_json),
            ],
        )

        require_equal(
            json.loads(captured_json.read_text(encoding="utf-8")),
            json.loads(captured_path.read_text(encoding="utf-8")),
            "CLI reporting must preserve the captured result document",
        )
        comparison = json.loads(comparison_json.read_text(encoding="utf-8"))
        require_equal(
            comparison["compatibility"]["comparable"],
            True,
            "captured and fresh results must be comparable in the CLI",
        )
        require_equal(
            comparison["regressions"],
            [],
            "the CLI gate must report no regression against the captured baseline",
        )
        require_equal(
            json.loads(judged_json.read_text(encoding="utf-8")),
            json.loads(judged_path.read_text(encoding="utf-8")),
            "CLI reporting must preserve the structured judged result document",
        )
        judged_comparison = json.loads(judged_comparison_json.read_text(encoding="utf-8"))
        require_equal(
            judged_comparison["structured_judge_comparison_state"],
            "compared",
            "the CLI comparison must preserve exact structured-judge semantics",
        )
        for report in (captured_html, fresh_html, judged_html, comparison_html):
            require(
                "<!doctype html>" in report.read_text(encoding="utf-8").lower(),
                "CLI HTML reports must be standalone documents",
            )
        judged_html_text = judged_html.read_text(encoding="utf-8")
        require(
            "correctness" in judged_html_text and "judge_output" not in judged_html_text,
            "the judged HTML report must remain explainable without raw provider output",
        )
        return {"status": "passed", "regressions": 0, "reports": 8}


def _provider_name() -> str:
    requested = os.environ.get("CAYU_PROVIDER")
    if requested is not None:
        requested = requested.strip().lower()
        if requested in {"openai", "anthropic"}:
            return requested
        raise SystemExit("CAYU_PROVIDER must be openai or anthropic.")
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "openai"


def _model(provider_name: str) -> str:
    if provider_name == "openai":
        return os.environ.get("CAYU_OPENAI_MODEL", "gpt-5.6-luna")
    return os.environ.get("CAYU_ANTHROPIC_MODEL", "claude-sonnet-4-6")


def _require_api_key(provider_name: str) -> None:
    if provider_name == "openai" and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY or choose CAYU_PROVIDER=anthropic.")
    if provider_name == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY or choose CAYU_PROVIDER=openai.")


def _record_browser_failures(
    page: Page,
    failures: dict[str, list[str]],
    *,
    deferred_source_run_aborts: list[str],
) -> None:
    page.on(
        "console",
        lambda message: (
            failures["console"].append(message.text) if message.type == "error" else None
        ),
    )
    page.on("pageerror", lambda error: failures["page"].append(str(error)))

    def record_request_failure(request) -> None:
        path = urlsplit(request.url).path
        mutation_id = request.headers.get("cayu-mutation-id")
        detail = f"{request.method} {request.url}: {request.failure}"
        if _is_deferred_source_run_abort(
            method=request.method,
            path=path,
            failure=request.failure,
        ):
            deferred_source_run_aborts.append(detail)
            return
        if _is_expected_browser_abort(
            method=request.method,
            path=path,
            failure=request.failure,
            mutation_id=mutation_id,
        ):
            return
        failures["request"].append(detail)

    page.on("requestfailed", record_request_failure)

    async def record_api_response(response) -> None:
        if "/api/" in response.url and response.status >= 400:
            failures["api"].append(f"{response.status} {response.request.method} {response.url}")

    page.on("response", record_api_response)


def _is_expected_browser_abort(
    *,
    method: str,
    path: str,
    failure: str | None,
    mutation_id: str | None,
) -> bool:
    return is_superseded_browser_read_abort(
        method=method,
        path=path,
        failure=failure,
        mutation_id=mutation_id,
    )


def _is_deferred_source_run_abort(
    *,
    method: str,
    path: str,
    failure: str | None,
) -> bool:
    return method == "POST" and path == "/api/run" and failure == "net::ERR_ABORTED"


def _require_no_browser_failures(
    failures: dict[str, list[str]],
    *,
    deferred_source_run_aborts: list[str],
    source_session_terminal: bool,
) -> None:
    require(
        not deferred_source_run_aborts or source_session_terminal,
        "the source run observer aborted without a proven terminal source session",
    )
    require(
        len(deferred_source_run_aborts) <= 1,
        "the source run produced more than one deferred observer abort",
    )
    observed = {name: entries for name, entries in failures.items() if entries}
    require_equal(
        observed, {}, "the real eval browser journey must have no browser or API failures"
    )


def _unused_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_for_server(
    process: subprocess.Popen[str],
    base_url: str,
    log_path: Path,
) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        status = process.poll()
        if status is not None:
            raise AssertionError(
                f"generated-project server exited {status} during startup:\n{_read_log(log_path)}"
            )
        try:
            with urlopen(f"{base_url}/cayu/", timeout=0.5) as response:
                if response.status == 200:
                    return
        except (OSError, URLError):
            pass
        time.sleep(0.05)
    raise TimeoutError(f"generated-project server did not start:\n{_read_log(log_path)}")


def _stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _read_log(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return f"<could not read server log: {exc}>"
    return text[-8_000:]


if __name__ == "__main__":
    asyncio.run(main())
