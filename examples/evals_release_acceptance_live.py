"""Credential-gated release acceptance for the dashboard-to-CI eval journey.

OpenAI:
    CAYU_PROVIDER=openai uv run python examples/evals_release_acceptance_live.py

Anthropic:
    CAYU_PROVIDER=anthropic uv run python examples/evals_release_acceptance_live.py

The check performs four agent executions: one source session, two durable
dashboard executions, and one local CLI rerun. Provider-level retry behavior is
left to the selected provider implementation.
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
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import uvicorn
from playwright.async_api import Page, async_playwright, expect

from _live_checks import require, require_equal, require_successful_terminal
from cayu import AgentSpec, CayuApp, CorpusTarget, EvalPlan, Message, RunRequest, SQLiteEvalStore
from cayu.providers import AnthropicProvider, ModelProvider, OpenAIProvider
from cayu.server import (
    BasicAuth,
    DashboardConfig,
    EvalsConfig,
    EvaluationPromotionConfig,
    ServerConfig,
    create_server,
)

AGENT_NAME = "eval-release-acceptance"
AUTH_PASSWORD = "eval-release-acceptance-password"
AUTH_USERNAME = "eval-release-acceptance-operator"
EVIDENCE_PREFIX = "CAYU_NIGHTLY_EVIDENCE="
EXPECTED_MARKER = "CAYU_EVAL_RELEASE_ACCEPTANCE_OK"
PROMOTION_SESSION_ID = "eval-release-acceptance-source"
TARGET_KEY = "release.acceptance"


def build_eval_plan() -> EvalPlan:
    """Build the local rerun target loaded by ``cayu eval run``."""

    provider_name = _provider_name()
    _require_api_key(provider_name)
    app = _application(provider_name)
    return EvalPlan(
        corpus_target=CorpusTarget(
            key=TARGET_KEY,
            app=app,
            request_base=RunRequest(agent_name=AGENT_NAME, messages=[]),
            application_release_id=f"{provider_name}-local-ci-release",
        )
    )


async def main() -> None:
    provider_name = _provider_name()
    _require_api_key(provider_name)
    app = _application(provider_name)
    source_events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name=AGENT_NAME,
                session_id=PROMOTION_SESSION_ID,
                max_steps=1,
                messages=[
                    Message.text(
                        "user",
                        f"Reply with exactly {EXPECTED_MARKER} and no other text.",
                    )
                ],
            )
        )
    ]
    require_successful_terminal(source_events)

    evals_directory = tempfile.TemporaryDirectory(prefix="cayu-evals-release-live-")
    eval_store = SQLiteEvalStore(Path(evals_directory.name) / "evals.sqlite")
    server_app = create_server(
        app,
        config=ServerConfig.protected(
            BasicAuth(username=AUTH_USERNAME, password=AUTH_PASSWORD),
            dashboard=DashboardConfig(),
            evaluation_promotion=EvaluationPromotionConfig(
                target_key=TARGET_KEY,
                source_agent_name=AGENT_NAME,
                application_release_id=f"{provider_name}-dashboard-release",
            ),
            evals=EvalsConfig(
                target=CorpusTarget(
                    key=TARGET_KEY,
                    app=app,
                    request_base=RunRequest(agent_name=AGENT_NAME, messages=[]),
                    application_release_id=f"{provider_name}-dashboard-release",
                ),
                store=eval_store,
                poll_interval_seconds=0.05,
            ),
        ),
    )
    listener = _loopback_listener()
    base_url = f"http://127.0.0.1:{listener.getsockname()[1]}"
    server = uvicorn.Server(uvicorn.Config(server_app, lifespan="on", log_level="warning"))
    server_task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        await _wait_for_server(server, server_task)
        evidence = await _run_browser_journey(base_url)
        evidence.update(
            {
                "provider": provider_name,
                "model": _model(provider_name),
                "source_session_completed": True,
            }
        )
        print(f"{EVIDENCE_PREFIX}{json.dumps(evidence, sort_keys=True)}")
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(server_task, timeout=10)
        except TimeoutError:
            server_task.cancel()
            await asyncio.gather(server_task, return_exceptions=True)
        await eval_store.close()
        evals_directory.cleanup()


async def _run_browser_journey(base_url: str) -> dict[str, object]:
    failures: dict[str, list[str]] = {
        "console": [],
        "page": [],
        "request": [],
        "api": [],
    }
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context(
            http_credentials={"username": AUTH_USERNAME, "password": AUTH_PASSWORD},
            viewport={"width": 1440, "height": 1000},
            accept_downloads=True,
        )
        page = await context.new_page()
        _record_browser_failures(page, failures)
        try:
            await page.goto(
                f"{base_url}/cayu/sessions/{PROMOTION_SESSION_ID}",
                wait_until="networkidle",
            )
            await page.get_by_test_id("promote-to-eval").click()
            sheet = page.get_by_test_id("promotion-sheet")
            await expect(sheet).to_be_visible()

            await sheet.get_by_label("Case name", exact=True).fill(
                "Real-provider release regression"
            )
            await sheet.get_by_role("button", name="Add assertion", exact=True).click()
            output_assertion = sheet.get_by_test_id("promotion-assertion").nth(1)
            await output_assertion.get_by_label("Assertion ID", exact=True).fill(
                "output-contains-marker"
            )
            await output_assertion.get_by_label("Type", exact=True).select_option(
                "final_output_contains"
            )
            await output_assertion.get_by_label("Expected output text", exact=True).fill(
                EXPECTED_MARKER
            )
            await sheet.get_by_test_id("promotion-preview").click()
            await expect(
                sheet.get_by_text("This score matches the current edits.", exact=True)
            ).to_be_visible(timeout=20_000)
            await expect(sheet.get_by_test_id("promotion-export")).to_be_enabled()

            async with page.expect_download() as corpus_download_info:
                await sheet.get_by_test_id("promotion-export").click()
            corpus_download = await corpus_download_info.value
            corpus_path_text = await corpus_download.path()
            require(corpus_path_text is not None, "promotion must download a corpus")
            corpus_path = Path(corpus_path_text)
            corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
            require_equal(corpus["target_key"], TARGET_KEY, "promotion target must be portable")
            require(
                PROMOTION_SESSION_ID not in json.dumps(corpus, sort_keys=True),
                "the promoted corpus must not expose the source session ID",
            )
            require_equal(
                [assertion["kind"] for assertion in corpus["cases"][0]["assertions"]],
                ["root_status", "final_output_contains"],
                "the exported case must retain both reviewed assertions",
            )

            await sheet.get_by_test_id("promotion-save").click()
            await expect(
                sheet.get_by_text(re.compile(r"Saved corpus .* to Evals\."))
            ).to_be_visible()
            await sheet.get_by_role("link", name="Open Evals", exact=True).click()
            await expect(page.get_by_role("heading", name="Evals", exact=True)).to_be_visible()

            suite_name = corpus["suites"][0]["name"]
            suite_id = corpus["suites"][0]["id"]
            baseline_run_id = await _launch_suite(page, suite_name, suite_id)
            await _require_completed_result(page)
            await expect(page.locator("pre").filter(has_text=EXPECTED_MARKER)).to_be_visible()
            baseline_path = await _download_result(page, baseline_run_id)
            await _download_html_report(page, baseline_run_id)

            await _reopen_catalog(page, suite_name)
            current_run_id = await _launch_suite(page, suite_name, suite_id)
            require(current_run_id != baseline_run_id, "dashboard runs must have distinct IDs")
            await _require_completed_result(page)
            await page.get_by_label("Baseline run ID", exact=True).fill(baseline_run_id)
            await page.get_by_role("button", name="Compare", exact=True).click()
            await expect(page.get_by_text("These runs are comparable.", exact=True)).to_be_visible()
            await expect(
                page.get_by_text("No compatible-result regressions.", exact=True)
            ).to_be_visible()

            local_evidence = await _run_local_ci_gate(corpus_path, baseline_path)
            _require_no_browser_failures(failures)
            return {
                "browser": "chromium",
                "dashboard_runs": 2,
                "assertions": 2,
                "comparison_regressions": 0,
                "local_ci": local_evidence,
            }
        finally:
            await browser.close()


async def _launch_suite(page: Page, suite_name: str, suite_id: str) -> str:
    await page.get_by_role(
        "button",
        name=f"Run suite {suite_name} ({suite_id})",
        exact=True,
    ).click()
    await expect(page).to_have_url(re.compile(r"[?&]tab=runs(?:&|$)"))
    run_ids = parse_qs(urlsplit(page.url).query).get("run", [])
    require_equal(len(run_ids), 1, "dashboard launch must select one durable run")
    return run_ids[0]


async def _require_completed_result(page: Page) -> None:
    published = page.locator('[data-slot="card-title"]').filter(has_text="Published result")
    await expect(published).to_be_visible(timeout=330_000)
    await expect(page.get_by_text("passed", exact=True).first).to_be_visible()
    await expect(page.get_by_text("session-completed", exact=True)).to_be_visible()
    output_assertion = page.get_by_role("row").filter(has_text="output-contains-marker")
    await expect(output_assertion).to_have_count(1)
    await expect(output_assertion.get_by_text("passed", exact=True)).to_be_visible()


async def _download_result(page: Page, run_id: str) -> Path:
    async with page.expect_download() as download_info:
        await page.get_by_role("button", name="JSON", exact=True).click()
    download = await download_info.value
    require_equal(
        download.suggested_filename,
        f"{run_id}.eval-result.json",
        "dashboard JSON must retain its server-owned filename",
    )
    path = await download.path()
    require(path is not None, "dashboard result download must be readable")
    return Path(path)


async def _download_html_report(page: Page, run_id: str) -> None:
    async with page.expect_download() as download_info:
        await page.get_by_role("button", name="HTML", exact=True).click()
    download = await download_info.value
    require_equal(
        download.suggested_filename,
        f"{run_id}.eval-report.html",
        "dashboard HTML must retain its server-owned filename",
    )


async def _reopen_catalog(page: Page, suite_name: str) -> None:
    await page.get_by_role("tab", name="Catalog", exact=True).click()
    await expect(page.get_by_role("button", name=suite_name, exact=True)).to_be_visible()


async def _run_local_ci_gate(corpus_path: Path, baseline_path: Path) -> dict[str, object]:
    environment = os.environ.copy()
    application_root = Path(__file__).resolve().parent

    def run(arguments: list[str]) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "cayu", "eval", *arguments],
            cwd=application_root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=360,
            check=False,
        )
        if completed.returncode != 0:
            raise AssertionError(
                f"local eval command exited {completed.returncode}: "
                f"{completed.stdout}{completed.stderr}"
            )

    with tempfile.TemporaryDirectory(prefix="cayu-evals-release-local-") as temporary:
        root = Path(temporary)
        local_result = root / "local-result.json"
        local_report = root / "local-report.html"
        comparison_path = root / "comparison.json"
        await asyncio.to_thread(run, ["validate", str(corpus_path)])
        await asyncio.to_thread(
            run,
            [
                "run",
                "evals_release_acceptance_live:build_eval_plan",
                "--corpus",
                str(corpus_path),
                "--output",
                str(local_result),
                "--html-output",
                str(local_report),
            ],
        )
        await asyncio.to_thread(
            run,
            [
                "compare",
                str(baseline_path),
                str(local_result),
                "--json",
                "--output",
                str(comparison_path),
            ],
        )
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        require_equal(
            comparison["compatibility"]["comparable"],
            True,
            "dashboard and local real-provider runs must be comparable",
        )
        require_equal(
            comparison["regressions"],
            [],
            "the local real-provider rerun must pass its dashboard baseline",
        )
        require(
            comparison["baseline"]["application_release_id"]
            != comparison["current"]["application_release_id"],
            "the CI gate must compare distinct target releases",
        )
        return {"status": "passed", "regressions": 0}


def _application(provider_name: str) -> CayuApp:
    app = CayuApp(enable_logging=False)
    app.register_provider(_provider(provider_name), default=True)
    app.register_agent(AgentSpec(name=AGENT_NAME, model=_model(provider_name)))
    return app


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


def _provider(provider_name: str) -> ModelProvider:
    if provider_name == "openai":
        return OpenAIProvider()
    return AnthropicProvider()


def _require_api_key(provider_name: str) -> None:
    if provider_name == "openai" and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY or choose CAYU_PROVIDER=anthropic.")
    if provider_name == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY or choose CAYU_PROVIDER=openai.")


def _record_browser_failures(page: Page, failures: dict[str, list[str]]) -> None:
    page.on(
        "console",
        lambda message: (
            failures["console"].append(message.text) if message.type == "error" else None
        ),
    )
    page.on("pageerror", lambda error: failures["page"].append(str(error)))
    page.on(
        "requestfailed",
        lambda request: failures["request"].append(
            f"{request.method} {request.url}: {request.failure}"
        ),
    )

    async def record_api_response(response) -> None:
        if "/api/" in response.url and response.status >= 400:
            failures["api"].append(f"{response.status} {response.request.method} {response.url}")

    page.on("response", record_api_response)


def _require_no_browser_failures(failures: dict[str, list[str]]) -> None:
    observed = {name: entries for name, entries in failures.items() if entries}
    require_equal(
        observed, {}, "the real eval browser journey must have no browser or API failures"
    )


def _loopback_listener() -> socket.socket:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    listener.setblocking(False)
    return listener


async def _wait_for_server(server: uvicorn.Server, server_task: asyncio.Task[None]) -> None:
    for _ in range(200):
        if server.started:
            return
        if server_task.done():
            await server_task
            raise AssertionError("eval release acceptance server stopped during startup")
        await asyncio.sleep(0.025)
    raise TimeoutError("eval release acceptance server did not start")


if __name__ == "__main__":
    asyncio.run(main())
