"""Opt-in browser regression for the prototype's native artifact links."""

from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("CAYU_DASHBOARD_PROTOTYPE_LIVE") != "1",
    reason="Opt-in prototype browser acceptance (requires dashboard npm ci and Chromium).",
)


@pytest.fixture(scope="module")
def prototype_url(tmp_path_factory):
    root = Path(__file__).resolve().parents[2]
    dashboard = root / "dashboard"
    node = shutil.which("node")
    assert node is not None, "Install Node >=22.18 before running prototype acceptance."
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    url = f"http://127.0.0.1:{port}/prototype.html"
    log_path = tmp_path_factory.mktemp("prototype-vite") / "server.log"
    with log_path.open("w+") as log:
        server = subprocess.Popen(
            [
                node,
                str(dashboard / "node_modules/vite/bin/vite.js"),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--strictPort",
            ],
            cwd=dashboard,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                assert server.poll() is None, log_path.read_text()
                try:
                    with urlopen(url, timeout=1) as response:
                        if response.status == 200:
                            break
                except (URLError, TimeoutError):
                    time.sleep(0.1)
            else:
                pytest.fail(f"Prototype server did not start: {log_path.read_text()}")
            yield url
        finally:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)


def test_prototype_artifact_native_links_stay_inside_fake_data(prototype_url, tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(headless=True)
        try:
            context = browser.new_context(accept_downloads=True)
            requests: list[str] = []

            def refuse_real_api(route):
                requests.append(route.request.url)
                route.abort()

            context.route(re.compile(r"^https?://[^/]+/api/"), refuse_real_api)
            page = context.new_page()
            page.goto(f"{prototype_url}#/artifacts")
            raw = page.get_by_role("link", name="Open raw", exact=True)
            download = page.get_by_role("link", name="Download", exact=True)
            playwright.expect(raw).to_be_visible(timeout=15000)
            playwright.expect(download).to_be_visible()

            # Native links must already point at local data. This covers copying the
            # address and opening it from a context menu, which bypass click handlers.
            expected = page.evaluate(
                "async href => await (await fetch(href)).text()", raw.get_attribute("href")
            )
            assert expected.strip()
            for link in (raw, download):
                href = link.get_attribute("href")
                assert href is not None and href.startswith("blob:"), href
                other = context.new_page()
                other.goto(href)
                assert other.locator("body").inner_text().strip() == expected.strip()
                other.close()

            with context.expect_page() as opened:
                raw.click()
            opened.value.wait_for_load_state()
            assert opened.value.locator("body").inner_text().strip() == expected.strip()
            opened.value.close()

            with context.expect_page() as opened:
                raw.click(modifiers=["ControlOrMeta"])
            opened.value.wait_for_load_state()
            assert opened.value.locator("body").inner_text().strip() == expected.strip()
            opened.value.close()

            with page.expect_download() as downloaded:
                download.click()
            result = downloaded.value
            assert result.suggested_filename.endswith((".md", ".csv"))
            destination = tmp_path / "artifact"
            result.save_as(destination)
            assert destination.read_text() == expected

            with page.expect_download() as modified_download:
                download.click(modifiers=["ControlOrMeta"])
            modified_download.value.save_as(destination)
            assert destination.read_text() == expected
            assert requests == [], requests
        finally:
            browser.close()


def test_prototype_reset_refreshes_mounted_suite_list(prototype_url):
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(prototype_url + "#/evals")
            page.locator("#prototype-app").select_option("new-app")
            page.get_by_role("link", name="Evals", exact=True).click()
            page.get_by_role("button", name="New suite", exact=True).click()
            page.locator("#suite-name").fill("Reset regression suite")
            page.get_by_placeholder("Input, as the agent would receive it").fill("hello")
            page.get_by_placeholder("The final answer contains…").fill("hello")
            page.get_by_role("button", name="Create suite", exact=True).click()
            page.wait_for_url(re.compile(r"#/evals/suites/"))
            page.get_by_role("link", name="Evals", exact=True).click()
            suite = page.get_by_role("link", name="Reset regression suite", exact=True)
            playwright.expect(suite).to_be_visible()
            page.get_by_role("button", name="Reset data", exact=True).click()
            playwright.expect(suite).to_have_count(0)
            playwright.expect(page.get_by_text("No eval suites yet", exact=True)).to_be_visible()
        finally:
            browser.close()


def test_prototype_legacy_hash_filters_survive_links_edits_and_reload(prototype_url):
    from urllib.parse import parse_qs, urlsplit

    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(prototype_url + "#/environments")
            links = page.locator('a[href*="#/artifacts?"][href*="q="]')
            playwright.expect(links.first).to_be_visible()
            link = links.last
            target = link.get_attribute("href")
            assert target is not None
            expected = parse_qs(urlsplit(target).fragment.partition("?")[2])
            link.click()
            search = page.get_by_placeholder(
                "Filter loaded page by file, session, agent, environment..."
            )
            playwright.expect(search).to_have_value(expected["q"][0])
            playwright.expect(
                page.get_by_text("store: " + expected["artifact_store_id"][0], exact=True)
            ).to_be_visible()
            page.reload()
            playwright.expect(search).to_have_value(expected["q"][0])
            search.fill("edited-filter")
            page.get_by_role("button", name="Clear links", exact=True).click()
            page.wait_for_function("location.hash.includes('q=edited-filter')")
            page.reload()
            playwright.expect(search).to_have_value("edited-filter")
            page.goto("about:blank")
            page.goto(prototype_url + "#/artifacts?q=direct-filter&environment_name=direct-env")
            playwright.expect(search).to_have_value("direct-filter")
            playwright.expect(
                page.get_by_text("environment: direct-env", exact=True)
            ).to_be_visible()
        finally:
            browser.close()


def test_prototype_created_task_tracks_session_and_preserves_explicit_holds(prototype_url):
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(prototype_url)
            observed = page.evaluate("""async () => {
              const {createMockApi} = await import('/src/next/mock/api.ts');
              const api = createMockApi('month-end');
              const before = await api.tasks();
              const held = before.find(t => ['pending', 'blocked', 'paused'].includes(t.status));
              if (!held) throw new Error('Seed needs an explicit holdable task');
              await api.holdTask(held.id, 'pause', 'Keep this hold');
              const made = await api.createTask({type:'reconcile_accruals',input:{period:'2099-01'}});
              const initial = await api.task(made.task_id);
              await new Promise(resolve => setTimeout(resolve, 2400));
              const completed = await api.task(made.task_id);
              const heldAfter = await api.task(held.id);
              await api.resumeTask(held.id);
              const resumed = await api.task(held.id);
              const again = await api.runAgain(made.session_id);
              const repeated = (await api.tasks()).find(t => t.session_id === again.session_id);
              await new Promise(resolve => setTimeout(resolve, 2400));
              const repeatedDone = await api.task(repeated.id);
              const resetting = await api.createTask({type:'reconcile_accruals',input:{period:'2099-02'}});
              api.reset();
              const resetIds = (await api.tasks()).map(t => t.id);
              await new Promise(resolve => setTimeout(resolve, 2400));
              const afterReset = (await api.tasks()).map(t => t.id);
              return {initial,completed,heldAfter,resumed,repeated,repeatedDone,resetting,resetIds,afterReset};
            }""")
            assert observed["initial"]["status"] == "running"
            assert observed["completed"]["status"] == "completed"
            assert observed["completed"]["completed_at"] is not None
            assert observed["completed"]["result"]["reply"]
            assert observed["heldAfter"]["status"] == "paused"
            assert observed["heldAfter"]["status_reason"] == "Keep this hold"
            assert observed["resumed"]["status"] != "paused"
            assert observed["repeated"]["status"] == "running"
            assert observed["repeatedDone"]["status"] == "completed"
            assert observed["repeatedDone"]["input"] == observed["completed"]["input"]
            assert observed["resetting"]["task_id"] not in observed["afterReset"]
            assert observed["resetIds"] == observed["afterReset"]
        finally:
            browser.close()
