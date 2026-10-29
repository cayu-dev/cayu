"""Production adoption of the four remaining pages with real signed roles and fake data."""

import asyncio
import os
import re
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urljoin, urlsplit

import pytest

pytest.importorskip("playwright.async_api")
from playwright.async_api import expect
from tests.server.test_dashboard_home_inbox import _home_browser
from tests.server.test_server_inbox import _app

from cayu.artifacts import ArtifactScope
from cayu.artifacts.local import LocalArtifactStore
from cayu.environments import Environment, EnvironmentSpec

pytestmark = pytest.mark.skipif(
    os.environ.get("CAYU_DASHBOARD_REMAINING_LIVE") != "1",
    reason="Opt-in remaining-pages Chromium acceptance.",
)


@pytest.mark.parametrize("role,nested,mobile", [("admin", False, False), ("member", True, True)])
def test_remaining_pages_roles_links_and_layout(tmp_path, role, nested, mobile):
    async def exercise():
        runtime = _app()
        store = LocalArtifactStore(tmp_path / "files", store_id="browser-files")
        name = "Review / λ space"
        runtime.register_environment(
            Environment(EnvironmentSpec(name=name), artifact_store=store), default=True
        )
        artifact = await store.put_bytes(
            b"Made-up deployment report\nAll checks passed.\n",
            filename="review-report.txt",
            content_type="text/plain",
            scope=ArtifactScope.ENVIRONMENT,
            environment_name=name,
        )
        async with _home_browser(
            role=role, nested=nested, mobile=mobile, seeded_runtime=runtime
        ) as (page, base, requests, responses):
            base = f"{base}/cayu"
            errors = []
            captures = Path(__file__).resolve().parents[2] / ".agents" / "screenshots"
            captures.mkdir(parents=True, exist_ok=True)
            page.on("pageerror", lambda error: errors.append(str(error)))
            await page.goto(f"{base}/environments", wait_until="networkidle")
            await page.get_by_role("link", name=name, exact=True).click()
            await expect(page.get_by_text("Commands run in", exact=True)).to_be_visible()
            await expect(page.get_by_text("No runner", exact=True).first).to_be_visible()
            await expect(
                page.get_by_text("This environment has no workspace.", exact=True)
            ).to_be_visible()
            await expect(
                page.get_by_role("link", name=re.compile(r"^review-report\.txt"))
            ).to_be_visible()
            await page.screenshot(path=str(captures / f"environment-{role}.png"))
            await page.get_by_role("link", name=re.compile(r"^review-report\.txt")).click()
            dialog = page.get_by_role("dialog")
            await expect(
                dialog.get_by_text("Made-up deployment report", exact=False)
            ).to_be_visible()
            query = parse_qs(urlsplit(page.url).query)
            assert query["artifact"] == [artifact.id]
            assert query["preview_store_id"] == [store.id]
            download = dialog.get_by_role("link", name="Download", exact=True)
            href = await download.get_attribute("href")
            content = await page.request.get(urljoin(page.url, href))
            assert content.status == 200
            assert await content.body() == b"Made-up deployment report\nAll checks passed.\n"
            await expect(dialog.get_by_role("link", name="Open raw", exact=True)).to_be_visible()
            assert await dialog.evaluate("el => el.getBoundingClientRect().right <= innerWidth + 1")
            await page.screenshot(path=str(captures / f"artifact-{role}.png"))
            await page.keyboard.press("Escape")
            await expect(dialog).to_have_count(0)
            await page.reload(wait_until="networkidle")
            assert "artifact" not in parse_qs(urlsplit(page.url).query)
            await page.goto(f"{base}/usage", wait_until="networkidle")
            await expect(
                page.get_by_role("button", name="Last 30 days", exact=True)
            ).to_be_visible()
            await expect(page.get_by_text("Sessions per day", exact=True)).to_be_visible()
            await page.get_by_role("button", name="Last 7 days", exact=True).click()
            await expect(page).to_have_url(re.compile(r"[?&]range=7d(?:&|$)"))
            start = "2026-07-01T00:00:59.999123Z"
            end = "2026-07-02T08:30:00.123456+06:00"
            search = urlencode(
                [
                    ("range", "custom"),
                    ("start_at", start),
                    ("end_at", end),
                    ("label", "team=qa"),
                    ("label", "kind=demo"),
                ]
            )
            async with page.expect_request(
                lambda req: req.method == "POST" and req.url.endswith("/api/usage/rollup")
            ) as pending:
                await page.goto(f"{base}/usage?{search}", wait_until="networkidle")
            body = (await pending.value).post_data_json
            assert body["start_at"] == start and body["end_at"] == end
            async with page.expect_request(
                lambda req: req.method == "POST" and req.url.endswith("/api/usage/rollup")
            ) as pending:
                await page.get_by_label("Specialist", exact=True).select_option("asker")
            body = (await pending.value).post_data_json
            assert body["start_at"] == start and body["end_at"] == end
            assert parse_qs(urlsplit(page.url).query)["label"] == ["team=qa", "kind=demo"]
            await page.screenshot(path=str(captures / f"usage-{role}.png"))
            # Invalid custom bounds explain the error without dispatching a usage request.
            requests.clear()
            invalid = urlencode(
                {
                    "range": "custom",
                    "start_at": "2026-02-30T00:00:00Z",
                    "end_at": "2026-03-03T00:00:00Z",
                }
            )
            await page.goto(f"{base}/usage?{invalid}", wait_until="networkidle")
            await expect(page.get_by_role("alert").first).to_be_visible()
            assert not any("/api/usage/rollup" in path for path in requests)
            requests.clear()
            await page.goto(f"{base}/system", wait_until="networkidle")
            if role == "member":
                await expect(page.get_by_test_id("dashboard-role-unavailable")).to_be_visible()
                assert not any("/api/system" in path for path in requests)
            else:
                await expect(page.get_by_text("About this agent", exact=True)).to_be_visible()
                assert any("/api/system" in path for path in requests)
            assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            assert not errors
            assert not [(r.status, r.url) for r in responses if r.status >= 400]

    asyncio.run(exercise())


def test_usage_live_journey_preserves_pricing_refresh_and_superseded_queries(monkeypatch):
    import importlib

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples"))
    journey = importlib.import_module("dashboard_behavior_live")

    async def exercise():
        runtime, _provider, _store, _tasks, directory = await journey._seed_app()
        try:
            async with _home_browser(
                seeded_runtime=runtime,
                dashboard_config={
                    "priceBook": journey._dashboard_price_book().model_dump(mode="json")
                },
            ) as (page, base, _requests, _responses):
                page.set_default_timeout(30_000)
                await journey._exercise_usage_page(page, base)
        finally:
            directory.cleanup()

    asyncio.run(exercise())


def test_artifact_exact_store_paging_and_disabled_optional_reads(tmp_path):
    async def exercise():
        runtime = _app()
        first = LocalArtifactStore(tmp_path / "first", store_id="first-store")
        second = LocalArtifactStore(tmp_path / "second", store_id="second-store")
        for name, store in [("first", first), ("second", second)]:
            runtime.register_environment(
                Environment(EnvironmentSpec(name=name), artifact_store=store)
            )
            await store.put_bytes(
                f"Correct content from {name}".encode(),
                artifact_id="art_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                filename=f"{name}.txt",
                content_type="text/plain",
                scope=ArtifactScope.ENVIRONMENT,
                environment_name=name,
            )
        for index in range(26):
            await first.put_bytes(
                b"made-up",
                filename=f"page-file-{index}.txt",
                scope=ArtifactScope.ENVIRONMENT,
                environment_name="first",
            )
        async with _home_browser(role="member", nested=True, seeded_runtime=runtime) as (
            page,
            base,
            requests,
            responses,
        ):

            async def disable_optional(route):
                response = await route.fetch()
                body = await response.json()
                body["capabilities"]["surfaces"]["session_overview_queries"]["read"] = {
                    "enabled": False,
                    "unavailable_reason": "unsupported",
                }
                await route.fulfill(response=response, json=body)

            await page.route("**/api/contract", disable_optional)
            await page.goto(
                f"{base}/cayu/artifacts?artifact=art_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa&artifact_store_id=first-store",
                wait_until="networkidle",
            )
            dialog = page.get_by_role("dialog")
            await expect(
                dialog.get_by_text("Correct content from first", exact=True)
            ).to_be_visible()
            assert not any("/sessions/overview" in path for path in requests)
            await page.keyboard.press("Escape")
            await page.goto(
                f"{base}/cayu/artifacts?artifact_store_id=first-store&offset=25",
                wait_until="networkidle",
            )
            await expect(page.get_by_role("link", name="first.txt", exact=True)).to_be_visible()
            await expect(page.get_by_role("link", name="second.txt", exact=True)).to_have_count(0)
            await page.reload(wait_until="networkidle")
            assert parse_qs(urlsplit(page.url).query)["offset"] == ["25"]
            await page.goto(
                f"{base}/cayu/artifacts?artifact=art_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa&artifact_store_id=second-store",
                wait_until="networkidle",
            )
            await expect(
                dialog.get_by_text("Correct content from second", exact=True)
            ).to_be_visible()
            await page.keyboard.press("Escape")
            requests.clear()
            await page.goto(f"{base}/cayu/usage", wait_until="networkidle")
            await expect(
                page.get_by_text("Session counts are unavailable", exact=False)
            ).to_be_visible()
            assert not any("/sessions/overview" in path for path in requests)
            assert not [(r.status, r.url) for r in responses if r.status >= 400]

    asyncio.run(exercise())


@pytest.mark.parametrize("role", ["admin", "member"])
def test_existing_role_journey_remaining_reads(monkeypatch, role):
    import importlib

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples"))
    journey = importlib.import_module("dashboard_role_access")

    async def exercise():
        runtime, _tool, _sessions, _tasks = await journey._seed_app()
        async with _home_browser(role=role, seeded_runtime=runtime) as (
            page,
            base,
            requests,
            responses,
        ):
            await journey._exercise_remaining_reads(page, base)
            assert not any("/api/artifacts" in path for path in requests)
            assert not [(r.status, r.url) for r in responses if r.status >= 400]

    asyncio.run(exercise())


def test_artifact_preview_preserves_list_scope_and_second_page(tmp_path):
    async def exercise():
        runtime = _app()
        duplicate_id = "art_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        for name, count in (("second", 20), ("first", 26)):
            store = LocalArtifactStore(tmp_path / name, store_id=f"{name}-store")
            runtime.register_environment(
                Environment(EnvironmentSpec(name=name), artifact_store=store)
            )
            for index in range(count):
                await store.put_bytes(
                    f"Exact {name} content {index}".encode(),
                    artifact_id=duplicate_id if index == 0 else None,
                    filename=f"{name}-{index}.txt",
                    content_type="text/plain",
                    scope=ArtifactScope.ENVIRONMENT,
                    environment_name=name,
                )
        async with _home_browser(role="member", nested=True, seeded_runtime=runtime) as (
            page,
            base,
            _requests,
            responses,
        ):
            for store_filter, selected in ((None, "second"), ("first-store", "first")):
                query = {"offset": "25", "scope": "environment", "q": ".txt"}
                if store_filter:
                    query["artifact_store_id"] = store_filter
                await page.goto(
                    f"{base}/cayu/artifacts?{urlencode(query)}", wait_until="networkidle"
                )
                rows = page.locator("tbody tr")
                await expect(
                    page.get_by_role("link", name=f"{selected}-0.txt", exact=True)
                ).to_be_visible()
                before = await rows.all_text_contents()
                await page.get_by_role("link", name=f"{selected}-0.txt", exact=True).click()
                dialog = page.get_by_role("dialog")
                await expect(
                    dialog.get_by_text(f"Exact {selected} content 0", exact=True)
                ).to_be_visible()
                opened = parse_qs(urlsplit(page.url).query)
                assert opened.get("artifact_store_id") == ([store_filter] if store_filter else None)
                assert opened["preview_store_id"] == [f"{selected}-store"]
                assert opened["offset"] == ["25"]
                assert opened["artifact"] == [duplicate_id]
                assert await rows.all_text_contents() == before
                # A reload reconstructs the exact selection without changing list scope.
                await page.reload(wait_until="networkidle")
                await expect(
                    dialog.get_by_text(f"Exact {selected} content 0", exact=True)
                ).to_be_visible()
                await page.keyboard.press("Escape")
                await expect(dialog).to_have_count(0)
                assert parse_qs(urlsplit(page.url).query) == {k: [v] for k, v in query.items()}
                assert await rows.all_text_contents() == before
            # Existing direct bookmarks keep the store-scoped list after dismissal.
            await page.goto(
                f"{base}/cayu/artifacts?artifact={duplicate_id}&artifact_store_id=second-store",
                wait_until="networkidle",
            )
            await expect(
                page.get_by_role("dialog").get_by_text("Exact second content 0", exact=True)
            ).to_be_visible()
            await page.keyboard.press("Escape")
            assert parse_qs(urlsplit(page.url).query) == {"artifact_store_id": ["second-store"]}
            assert not [(r.status, r.url) for r in responses if r.status >= 400]

    asyncio.run(exercise())


def test_system_live_journey_keeps_manual_snapshot_refresh(monkeypatch):
    import importlib

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples"))
    journey = importlib.import_module("dashboard_behavior_live")

    async def exercise():
        async with _home_browser(seeded_runtime=_app()) as (page, base, _requests, _responses):
            await journey._exercise_system_page(page, base)

    asyncio.run(exercise())
