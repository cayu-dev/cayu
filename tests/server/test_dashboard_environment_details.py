"""Native environment inventory through the signed production dashboard."""

import asyncio
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit

import pytest

pytest.importorskip("playwright.async_api")
from playwright.async_api import expect
from tests.core.test_cloud_reporting import _app
from tests.server.test_dashboard_home_inbox import _home_browser

from cayu import RunRequest
from cayu.artifacts import ArtifactScope
from cayu.artifacts.local import LocalArtifactStore
from cayu.environments import Environment, EnvironmentSpec
from cayu.events import Event, EventType
from cayu.mcp.base import McpServerSpec
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionIdentity
from cayu.vaults.redaction import SecretRedactor

pytestmark = pytest.mark.skipif(
    os.environ.get("CAYU_DASHBOARD_ENVIRONMENT_LIVE") != "1",
    reason="Opt-in focused Environment inventory Chromium acceptance.",
)
NAME = "Review / λ space"
CANARY = "environment-private-canary"


async def _runtime(tmp_path, name=NAME):
    clock = datetime.now(UTC) - timedelta(days=3)
    store = InMemorySessionStore(ownership_clock=lambda: clock)
    app = _app(store, secret_redactor=SecretRedactor([CANARY]))
    files = LocalArtifactStore(tmp_path / "files", store_id="environment-demo-files")
    app.register_environment(
        Environment(
            EnvironmentSpec(name=name),
            artifact_store=files,
            mcp_servers=[
                McpServerSpec(name="Declared lookup", command=["not-executed"]),
                McpServerSpec(name=CANARY, command=[CANARY], env={"TOKEN": CANARY}),
            ],
        )
    )
    app.register_environment(Environment(EnvironmentSpec(name="Empty sandbox")))
    for session_id in ("first", "second"):
        await store.create(
            RunRequest(
                session_id=session_id,
                title=f"Review {session_id}",
                agent_name="writer",
                environment_name=name,
                messages=[],
            ),
            identity=SessionIdentity(provider_name="configured", model="configured-model"),
        )
    await store.append_event(
        "first",
        Event(session_id="first", type=EventType.MODEL_COMPLETED, timestamp=clock, payload={}),
    )
    original = await store.load("first")
    assert original is not None
    clock += timedelta(days=2)
    await store.update_metadata("first", {"note": "later metadata is not activity"})
    await files.put_bytes(
        b"Synthetic only",
        filename="environment-note.txt",
        content_type="text/plain",
        scope=ArtifactScope.ENVIRONMENT,
        environment_name=name,
    )
    return app, original.last_activity_at


@pytest.mark.parametrize("role", ["admin", "member"])
def test_environment_native_inventory_roles_metadata_and_links(tmp_path, role):
    async def exercise():
        app, activity = await _runtime(tmp_path)
        async with _home_browser(
            role=role, nested=True, mobile=role == "member", seeded_runtime=app
        ) as (page, base, _, _):
            requests = []
            page.on("request", lambda request: requests.append(request))
            await page.goto(f"{base}/cayu/environments", wait_until="networkidle")
            row = page.get_by_role("row").filter(
                has=page.get_by_role("link", name=NAME, exact=True)
            )
            await expect(row).to_contain_text("2 sessions")
            await expect(row).to_contain_text("Declared lookup")
            assert not [r for r in requests if "/sessions?" in r.url]
            await expect(
                page.get_by_text(
                    "Counts include all retained sessions, including eval trials and forks.",
                    exact=True,
                )
            ).to_be_visible()
            empty = page.get_by_role("row").filter(
                has=page.get_by_role("link", name="Empty sandbox", exact=True)
            )
            await expect(empty).to_contain_text("0 sessions")
            await expect(empty).to_contain_text("None yet")
            assert CANARY not in await page.locator("body").inner_text()
            assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            await page.get_by_role("link", name=NAME, exact=True).click()
            await expect(page.get_by_text("Local disk", exact=True)).to_be_visible()
            await expect(page.get_by_text("Declared lookup", exact=False).first).to_be_visible()
            await expect(page.get_by_text("2 sessions", exact=True)).to_be_visible()
            await expect(
                page.get_by_role("link", name="environment-note.txt", exact=False)
            ).to_be_visible()
            await page.get_by_text("Debug", exact=True).click()
            await expect(
                page.get_by_text(activity.isoformat().replace("+00:00", "Z"), exact=True).first
            ).to_be_visible()
            assert CANARY not in await page.locator("body").inner_text()
            await page.get_by_text("Debug", exact=True).click()
            assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            captures = Path(__file__).resolve().parents[2] / ".agents" / "screenshots"
            captures.mkdir(parents=True, exist_ok=True)
            await page.screenshot(
                path=str(captures / f"environment-native-{role}.png"), full_page=True
            )
            link = await page.get_by_role(
                "link", name="All sessions here", exact=True
            ).get_attribute("href")
            query = parse_qs(urlsplit(link).query)
            assert query["environment_name"] == [NAME]
            assert query["include_eval"] == ["true"] and query["include_forks"] == ["true"]
            # Existing name-search links keep working under the nested mount.
            await page.goto(
                f"{base}/cayu/environments?{urlencode({'q': NAME})}", wait_until="networkidle"
            )
            await expect(page.get_by_role("link", name="Empty sandbox", exact=True)).to_have_count(
                0
            )
            if role == "member":
                response = await page.request.get(f"{base}/api/system/diagnostics")
                assert response.status == 403

    asyncio.run(exercise())


@pytest.mark.parametrize("mode", ["unsupported", "missing-capability"])
def test_environment_declarations_remain_when_native_activity_is_unavailable(tmp_path, mode):
    async def exercise():
        app, _ = await _runtime(tmp_path)
        if mode == "unsupported":
            app.session_store.supports_environment_activity = False
        async with _home_browser(role="member", seeded_runtime=app) as (page, base, _, _):
            if mode == "missing-capability":

                async def hide_capability(route):
                    response = await route.fetch()
                    body = await response.json()
                    body["capabilities"]["surfaces"].pop("environment_details", None)
                    await route.fulfill(response=response, json=body)

                await page.route("**/api/contract", hide_capability)
            await page.goto(f"{base}/cayu/environments", wait_until="networkidle")
            row = page.get_by_role("row").filter(
                has=page.get_by_role("link", name=NAME, exact=True)
            )
            await expect(row).to_contain_text("Session count unavailable")
            await expect(row).to_contain_text("Declared lookup")
            await expect(row).not_to_contain_text("0 sessions")
            await page.get_by_role("link", name=NAME, exact=True).click()
            await expect(page.get_by_text("Local disk", exact=True)).to_be_visible()
            await expect(page.get_by_text("Declared lookup", exact=False).first).to_be_visible()
            await expect(page.get_by_text("2 sessions", exact=True)).to_have_count(0)

    asyncio.run(exercise())


def test_environment_refresh_retains_inventory_but_authorization_failure_hides_it(tmp_path):
    async def exercise():
        app, _ = await _runtime(tmp_path)
        async with _home_browser(seeded_runtime=app) as (page, base, _, _):
            await page.goto(f"{base}/cayu/environments", wait_until="networkidle")
            await page.get_by_role("link", name=NAME, exact=True).click()
            await expect(page.get_by_text("2 sessions", exact=True)).to_be_visible()
            status = 501

            async def failure(route):
                await route.fulfill(status=status, json={"detail": "Inventory unavailable."})

            await page.route("**/api/environments", failure)
            await page.get_by_role("button", name="Refresh", exact=True).click()
            await expect(
                page.get_by_text(
                    "Couldn't refresh environments; showing the last inventory loaded.", exact=True
                )
            ).to_be_visible(timeout=15000)
            await expect(page.get_by_text("2 sessions", exact=True)).to_be_visible()
            await expect(page.get_by_text("Declared lookup", exact=False).first).to_be_visible()
            await page.get_by_role("link", name="Environments", exact=True).last.click()
            await expect(page.get_by_text("2 sessions", exact=True)).to_be_visible()
            await expect(page.get_by_role("button", name="Refresh", exact=True)).to_be_enabled(
                timeout=15000
            )
            status = 403
            await page.get_by_role("button", name="Refresh", exact=True).click()
            await expect(page.get_by_role("link", name=NAME, exact=True)).to_have_count(
                0, timeout=15000
            )
            await expect(page.get_by_text("2 sessions", exact=True)).to_have_count(0)
            await expect(page.get_by_text("No environments", exact=True)).to_have_count(0)
            # A fresh failed read remains an error, not a zero or successful empty inventory.
            await page.reload(wait_until="networkidle")
            await expect(page.get_by_text("Inventory unavailable.", exact=True)).to_be_visible(
                timeout=15000
            )
            await expect(page.get_by_text("No environments", exact=True)).to_have_count(0)

    asyncio.run(exercise())


@pytest.mark.parametrize("name", [".", "..", "%2E", "%2E%2E", "Review / λ space"])
def test_exact_environment_names_survive_browser_navigation_and_reload(name):
    async def exercise():
        app = _app()
        for registered in (".", "..", "%2E", "%2E%2E", "Review / λ space"):
            app.register_environment(Environment(EnvironmentSpec(name=registered)))
        async with _home_browser(role="member", nested=True, seeded_runtime=app) as (
            page,
            base,
            _,
            _,
        ):
            await page.goto(f"{base}/cayu/environments", wait_until="networkidle")
            link = page.get_by_role("link", name=name, exact=True)
            href = await link.get_attribute("href")
            assert href is not None
            async with page.context.expect_page() as opened:
                await link.click(modifiers=["ControlOrMeta"])
            target = await opened.value
            try:
                await expect(target.get_by_role("heading", name=name, exact=True)).to_be_visible()
                await target.reload(wait_until="networkidle")
                await expect(target.get_by_role("heading", name=name, exact=True)).to_be_visible()
            finally:
                await target.close()
            await link.click()
            await expect(page.get_by_role("heading", name=name, exact=True)).to_be_visible()
            await page.reload(wait_until="networkidle")
            await expect(page.get_by_role("heading", name=name, exact=True)).to_be_visible()
            await (
                page.get_by_role("main")
                .get_by_role("link", name="Environments", exact=True)
                .click()
            )
            await expect(page.get_by_role("link", name=name, exact=True)).to_be_visible()
            assert "name" not in parse_qs(urlsplit(page.url).query)
            # Opening the copied href must retain its exact registered identity as well.
            await page.goto(
                f"{urlsplit(base).scheme}://{urlsplit(base).netloc}{href}", wait_until="networkidle"
            )
            await expect(page.get_by_role("heading", name=name, exact=True)).to_be_visible()

    asyncio.run(exercise())


def test_dot_environment_links_from_session_and_artifact_views(tmp_path):
    async def exercise():
        app, _ = await _runtime(tmp_path, name="..")
        async with _home_browser(role="member", nested=True, seeded_runtime=app) as (
            page,
            base,
            _,
            _,
        ):
            await page.goto(f"{base}/cayu/sessions/first", wait_until="networkidle")
            header_link = page.get_by_role("link", name="..", exact=True).first
            await expect(header_link).to_be_visible()
            header_href = await header_link.get_attribute("href")
            assert parse_qs(urlsplit(header_href).query)["name"] == [".."]
            await page.get_by_role("button", name="Session details", exact=True).click()
            links = page.get_by_role("link", name="..", exact=True)
            assert await links.count() >= 2
            for link in await links.all():
                assert await link.get_attribute("href") == header_href
            await links.last.click()
            await expect(page.get_by_role("heading", name="..", exact=True)).to_be_visible()
            await page.get_by_role("link", name="environment-note.txt", exact=False).click()
            drawer = page.get_by_role("dialog")
            artifact_link = drawer.get_by_role("link", name="..", exact=True)
            await expect(artifact_link).to_be_visible()
            assert await artifact_link.get_attribute("href") == header_href
            await artifact_link.click()
            await expect(page.get_by_role("heading", name="..", exact=True)).to_be_visible()
            await page.reload(wait_until="networkidle")
            await expect(page.get_by_role("heading", name="..", exact=True)).to_be_visible()

    asyncio.run(exercise())


@pytest.mark.parametrize("denied_status", [401, 403])
def test_environment_denial_cannot_be_replaced_by_a_transient_error(tmp_path, denied_status):
    async def exercise():
        app, _ = await _runtime(tmp_path)
        async with _home_browser(seeded_runtime=app) as (page, base, _, _):
            # Populate both the inventory and the previously separate Usage options cache.
            await page.goto(f"{base}/cayu/usage", wait_until="networkidle")
            await expect(page.locator("option").filter(has_text=NAME)).to_have_count(1)
            await page.get_by_role("link", name="Environments", exact=True).click()
            await page.get_by_role("link", name=NAME, exact=True).click()
            await expect(page.get_by_text("2 sessions", exact=True)).to_be_visible()
            status = denied_status

            async def failure(route):
                if status == 0:
                    await route.abort("failed")
                else:
                    await route.fulfill(status=status, json={"detail": "Inventory unavailable."})

            await page.route("**/api/environments", failure)
            refresh = page.get_by_role("button", name="Refresh", exact=True)
            await refresh.click()
            await expect(page.get_by_text("2 sessions", exact=True)).to_have_count(0, timeout=15000)
            await expect(refresh).to_be_enabled(timeout=15000)
            status = 503 if denied_status == 403 else 0
            await refresh.click()
            await expect(page.get_by_text("2 sessions", exact=True)).to_have_count(0)
            await expect(refresh).to_be_enabled(timeout=15000)
            await expect(page.get_by_text("2 sessions", exact=True)).to_have_count(0)
            await expect(page.get_by_text("Declared lookup", exact=False)).to_have_count(0)
            await expect(page.get_by_text("No environments", exact=True)).to_have_count(0)
            await page.get_by_role("link", name="Environments", exact=True).last.click()
            await expect(page.get_by_role("link", name=NAME, exact=True)).to_have_count(0)
            for destination in ("Artifacts", "Usage"):
                await page.get_by_role("link", name=destination, exact=True).click()
                await expect(page.locator("option").filter(has_text=NAME)).to_have_count(0)
            await page.get_by_role("link", name="Environments", exact=True).click()
            await expect(refresh).to_be_enabled(timeout=15000)
            await expect(page.get_by_role("link", name=NAME, exact=True)).to_have_count(0)
            await page.unroute("**/api/environments", failure)
            await refresh.click()
            await expect(page.get_by_role("link", name=NAME, exact=True)).to_be_visible()
            await page.get_by_role("link", name=NAME, exact=True).click()
            await expect(page.get_by_text("2 sessions", exact=True)).to_be_visible()

    asyncio.run(exercise())
