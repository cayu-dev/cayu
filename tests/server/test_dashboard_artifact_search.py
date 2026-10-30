"""Global artifact queries through the signed production page and exact preview links."""

import asyncio
import os
from urllib.parse import parse_qs, urlsplit

import pytest

pytest.importorskip("playwright.async_api")
from playwright.async_api import expect
from tests.core.test_cloud_reporting import _app
from tests.server.test_dashboard_home_inbox import _home_browser

from cayu import RunRequest
from cayu.artifacts import LocalArtifactStore
from cayu.environments import Environment, EnvironmentSpec
from cayu.sessions.records import SessionIdentity

pytestmark = pytest.mark.skipif(
    os.environ.get("CAYU_DASHBOARD_ARTIFACT_LIVE") != "1",
    reason="Opt-in focused artifact Chromium acceptance.",
)


async def _runtime(tmp_path):
    app = _app()
    await app.session_store.create(
        RunRequest(
            session_id="producer", agent_name="writer", title="Quarterly review", messages=[]
        ),
        identity=SessionIdentity(provider_name="p", model="m"),
    )
    for name in ("one", "two"):
        store = LocalArtifactStore(tmp_path / name, store_id=name)
        app.register_environment(Environment(EnvironmentSpec(name=name), artifact_store=store))
        for index in range(18):
            await store.put_bytes(
                f"Synthetic {name} {index}".encode(),
                artifact_id="art_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" if index == 0 else None,
                filename=f"{name}-{index:02}.txt",
                session_id="producer",
                agent_name="writer",
                environment_name=name,
            )
    return app


@pytest.mark.parametrize("role", ["admin", "member"])
def test_global_query_sort_paging_titles_and_preview_identity(tmp_path, role):
    async def exercise():
        async with _home_browser(
            role=role, nested=True, mobile=role == "member", seeded_runtime=await _runtime(tmp_path)
        ) as (page, base, _, _):
            requests = []
            page.on("request", lambda request: requests.append(request.url))
            await page.goto(f"{base}/cayu/artifacts", wait_until="networkidle")
            await expect(page.get_by_role("combobox", name="Sort files")).to_be_enabled()
            await page.get_by_role("combobox", name="Sort files").select_option("filename_asc")
            await expect(page.get_by_role("link", name="one-00.txt", exact=True)).to_be_visible()
            await expect(page.get_by_role("link", name="two-17.txt", exact=True)).to_have_count(0)
            field = page.get_by_role("textbox", name="Search files")
            await field.fill("two-17")
            await field.press("Enter")
            await expect(page.get_by_role("link", name="two-17.txt", exact=True)).to_be_visible()
            await expect(page.get_by_role("link", name="one-00.txt", exact=True)).to_have_count(0)
            assert parse_qs(urlsplit(page.url).query)["q"] == ["two-17"]
            await field.fill("Quarterly review")
            await field.press("Enter")
            await expect(page.get_by_role("link", name="one-00.txt", exact=True)).to_be_visible()
            assert not any("/sessions/overview" in url for url in requests)
            # Direct second-page bookmarks retain list scope while the exact duplicate ID previews another store.
            for store in (None, "one"):
                query = (
                    "order_by=filename_desc&offset=25"
                    if store is None
                    else "order_by=filename_desc&artifact_store_id=one"
                )
                await page.goto(f"{base}/cayu/artifacts?{query}", wait_until="networkidle")
                link = page.get_by_role("link", name="one-00.txt", exact=True)
                await expect(link).to_be_visible()
                before = parse_qs(urlsplit(page.url).query)
                await link.click()
                await expect(page.get_by_text("Synthetic one 0", exact=True)).to_be_visible()
                opened = parse_qs(urlsplit(page.url).query)
                assert opened["preview_store_id"] == ["one"]
                assert opened.get("artifact_store_id") == before.get("artifact_store_id")
                assert opened.get("offset") == before.get("offset")
                await page.keyboard.press("Escape")
                assert parse_qs(urlsplit(page.url).query) == before
                await expect(link).to_be_visible()

    asyncio.run(exercise())


@pytest.mark.parametrize("denial", [401, 403])
def test_artifact_inventory_denial_survives_transient_failure_and_navigation(tmp_path, denial):
    async def exercise():
        async with _home_browser(role="member", seeded_runtime=await _runtime(tmp_path)) as (
            page,
            base,
            _,
            _,
        ):
            mode = 200

            async def inventory(route):
                if mode == 200:
                    await route.continue_()
                else:
                    await route.fulfill(
                        status=mode, json={"detail": "Synthetic inventory read failed"}
                    )

            await page.route("**/api/artifacts?*", inventory)
            await page.goto(f"{base}/cayu/artifacts", wait_until="networkidle")
            marker = page.get_by_role("link", name="two-17.txt", exact=True)
            await expect(marker).to_be_visible()

            async def revisit():
                await (
                    page.get_by_role("navigation", name="Main")
                    .get_by_role("link", name="Home", exact=True)
                    .click()
                )
                await page.get_by_role("link", name="Artifacts", exact=True).click()
                await page.wait_for_load_state("networkidle")

            # A direct transient refresh may retain the last successful inventory.
            mode = 503
            await revisit()
            await expect(marker).to_be_visible()
            mode = denial
            await revisit()
            await expect(marker).to_have_count(0)
            mode = 503
            await revisit()
            await expect(marker).to_have_count(0)
            mode = 200
            await revisit()
            await expect(marker).to_be_visible()

    asyncio.run(exercise())


def test_preview_denial_discards_known_rows_and_cached_file_content(tmp_path):
    async def exercise():
        async with _home_browser(role="member", seeded_runtime=await _runtime(tmp_path)) as (
            page,
            base,
            _,
            _,
        ):
            await page.goto(f"{base}/cayu/artifacts", wait_until="networkidle")
            marker = page.get_by_role("link", name="two-17.txt", exact=True)
            await marker.click()
            await expect(
                page.get_by_role("dialog").get_by_text("Synthetic two 17", exact=True)
            ).to_be_visible()
            await page.keyboard.press("Escape")

            async def deny(route):
                await route.fulfill(status=403, json={"detail": "Synthetic preview denial"})

            await page.route("**/api/artifacts/*?*", deny)
            await marker.click()
            await expect(
                page.get_by_role("dialog").get_by_text("Synthetic two 17", exact=True)
            ).to_have_count(0)
            await expect(page.locator("tbody tr")).to_have_count(0)

    asyncio.run(exercise())
