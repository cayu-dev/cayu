"""Focused public-browser regressions for the remaining-page adoption review."""

import asyncio
import os
import re
from urllib.parse import parse_qs, quote, unquote, urlsplit

import pytest

pytest.importorskip("playwright.async_api")
from playwright.async_api import expect
from tests.server.test_dashboard_home_inbox import _home_browser
from tests.server.test_server_inbox import _app

from cayu.artifacts.local import LocalArtifactStore
from cayu.environments import Environment, EnvironmentSpec
from cayu.messages import Message
from cayu.sessions.requests import RunRequest
from cayu.sessions.store import SessionStore
from cayu.tasks import InMemoryTaskStore

pytestmark = pytest.mark.skipif(
    os.environ.get("CAYU_DASHBOARD_REMAINING_LIVE") != "1",
    reason="Opt-in adoption Chromium checks.",
)


async def _environment_app(tmp_path, *, session=False):
    runtime = _app(task_store=InMemoryTaskStore())
    name = "Review / λ space"
    store = LocalArtifactStore(tmp_path / "files", store_id="review-files")
    runtime.register_environment(
        Environment(EnvironmentSpec(name=name), artifact_store=store), default=True
    )
    runtime.register_environment(Environment(EnvironmentSpec(name="Other environment")))
    if session:
        async for _ in runtime.run(
            RunRequest(
                agent_name="approver",
                session_id="review-session",
                title="Review the report",
                messages=[Message.text("user", "go")],
            )
        ):
            pass
    return runtime, name


@pytest.mark.parametrize("overview", [False, True])
def test_environment_session_reads_keep_overview_optional(tmp_path, overview):
    async def exercise():
        runtime, name = await _environment_app(tmp_path, session=True)
        if not overview:

            class BasicStore(type(runtime.session_store)):
                invocation_lifecycle_command_version = 1
                terminal_interaction_publication_version = 1
                read_usage_accounting = SessionStore.read_usage_accounting

            runtime.session_store.__class__ = BasicStore
        async with _home_browser(role="member", nested=True, seeded_runtime=runtime) as (
            page,
            base,
            _,
            responses,
        ):
            requests = []
            page.on("request", lambda req: requests.append(req.url))
            await page.goto(f"{base}/cayu/environments", wait_until="networkidle")
            await page.get_by_role("link", name=name, exact=True).click()
            await expect(
                page.get_by_role(
                    "link",
                    name=re.compile("^Review the report" if overview else "^Untitled session"),
                )
            ).to_be_visible()
            relevant = [
                parse_qs(urlsplit(url).query)
                for url in requests
                if urlsplit(url).path.endswith("/api/sessions")
            ]
            assert relevant
            if not overview:
                assert all("overview" not in query.get("include", []) for query in relevant)
            else:
                assert sum(query.get("include") == ["overview"] for query in relevant) == 1
            assert not [(r.status, r.url) for r in responses if r.status >= 400]

    asyncio.run(exercise())


def test_environment_old_search_and_both_session_links(tmp_path):
    async def exercise():
        runtime, name = await _environment_app(tmp_path, session=True)
        async with _home_browser(nested=True, seeded_runtime=runtime) as (
            page,
            base,
            _,
            _responses,
        ):
            await page.goto(f"{base}/cayu/environments?q={quote(name)}", wait_until="networkidle")
            await expect(page.get_by_role("link", name=name, exact=True)).to_be_visible()
            await expect(
                page.get_by_role("link", name="Other environment", exact=True)
            ).to_have_count(0)
            await page.goto(f"{base}/cayu/sessions/review-session", wait_until="networkidle")

            async def exact_links():
                links = page.get_by_role("link", name=name, exact=True)
                assert await links.count() > 0
                for link in await links.all():
                    href = await link.get_attribute("href")
                    assert unquote(urlsplit(href).path) == f"/product/cayu/environments/{name}"
                    assert not urlsplit(href).query
                return links

            links = await exact_links()
            await page.get_by_role("button", name="Session details", exact=True).click()
            links = await exact_links()
            async with page.context.expect_page() as opened:
                await links.first.click(modifiers=["ControlOrMeta"])
            target = await opened.value
            await expect(target.get_by_text("Commands run in", exact=True)).to_be_visible()
            await target.close()
            await links.last.click()
            await expect(page.get_by_text("Commands run in", exact=True)).to_be_visible()

    asyncio.run(exercise())


def test_environment_failed_optional_reads_do_not_claim_empty_or_counts(tmp_path):
    async def exercise():
        runtime, name = await _environment_app(tmp_path)
        async with _home_browser(seeded_runtime=runtime) as (page, base, _, _responses):
            page.set_default_timeout(15_000)

            async def fail(route):
                await route.fulfill(
                    status=503,
                    content_type="application/json",
                    body='{"detail":"Temporary read failure"}',
                )

            await page.route("**/api/sessions?*", fail)
            await page.goto(f"{base}/cayu/environments", wait_until="networkidle")
            await expect(
                page.get_by_role("row").filter(has_text=name).get_by_text("Unavailable", exact=True)
            ).to_be_visible(timeout=15_000)
            await page.route("**/api/artifacts?*", fail)
            await page.get_by_role("link", name=name, exact=True).click()
            saved = page.get_by_text("Saved so far", exact=True).locator("..")
            await expect(saved).to_contain_text("Unavailable", timeout=15_000)
            await page.unroute("**/api/artifacts?*", fail)

            async def disable(route):
                response = await route.fetch()
                body = await response.json()
                body["capabilities"]["surfaces"]["artifacts"]["read"] = {
                    "enabled": False,
                    "unavailable_reason": "unsupported",
                }
                await route.fulfill(response=response, json=body)

            await page.route("**/api/contract", disable)
            await page.reload(wait_until="networkidle")
            await expect(saved).to_contain_text("Unavailable", timeout=15_000)
            await page.unroute("**/api/contract", disable)
            await page.unroute("**/api/sessions?*", fail)

            async def unknown_total(route):
                response = await route.fetch()
                body = await response.json()
                body["total_count"] = None
                await route.fulfill(response=response, json=body)

            await page.route("**/api/artifacts?*", unknown_total)
            await page.reload(wait_until="networkidle")
            await expect(saved).to_contain_text("Count unavailable")
            await page.unroute("**/api/artifacts?*", unknown_total)
            await page.reload(wait_until="networkidle")
            await expect(saved).to_contain_text("None yet")
            await page.route("**/api/artifacts?*", fail)
            await (
                page.get_by_role("main")
                .get_by_role("link", name="Environments", exact=True)
                .click()
            )
            await page.get_by_role("link", name=name, exact=True).click()
            await expect(saved).to_contain_text("last read; refresh failed", timeout=15_000)

    asyncio.run(exercise())


def test_system_graph_creation_uses_mutation_readiness(tmp_path):
    async def exercise():
        runtime, _ = await _environment_app(tmp_path)
        async with _home_browser(seeded_runtime=runtime) as (page, base, _, _responses):
            await page.goto(f"{base}/cayu/system", wait_until="networkidle")
            await expect(page.get_by_text("Task graph creation", exact=True)).to_have_count(0)

            async def disabled(route):
                response = await route.fetch()
                body = await response.json()
                body["capabilities"]["surfaces"]["task_graph_creation"]["mutate"] = {
                    "enabled": False,
                    "unavailable_reason": "not_configured",
                }
                await route.fulfill(response=response, json=body)

            await page.route("**/api/system/diagnostics", disabled)
            await page.reload(wait_until="networkidle")
            await expect(page.get_by_text("Task graph creation", exact=True)).to_be_visible()
            await expect(
                page.get_by_text("Task graph creation", exact=True).locator("..")
            ).to_contain_text("Not set up")

    asyncio.run(exercise())


def test_environment_branch_observations_preserve_terminal_and_incomplete_states(tmp_path):
    from tests.workspaces.test_local_workspace_branches import _created_branch, _source_and_request

    from cayu.workspaces import WorkspaceBranchPublicationRequest
    from cayu.workspaces.branches import WorkspaceBranchLifecycleSummary

    async def exercise():
        root = tmp_path / "workspace"
        root.mkdir()
        workspace, request = await _source_and_request(root)
        branch = await _created_branch(workspace, request)
        changes = await branch.changes()
        await branch.publish(
            WorkspaceBranchPublicationRequest(
                branch_id=branch.branch_id,
                baseline_revision=changes.baseline_revision,
                change_set_digest=changes.digest,
            )
        )
        assert workspace.branch_lifecycle_summary().statuses == ("committed",)
        runtime = _app()
        runtime.register_environment(
            Environment(EnvironmentSpec(name="branches"), workspace=workspace), default=True
        )
        async with _home_browser(seeded_runtime=runtime) as (page, base, _, _responses):
            await page.goto(f"{base}/cayu/environments/branches", wait_until="networkidle")
            row = page.get_by_text("Observed branches", exact=True).locator("..")
            await expect(row).to_contain_text("1 committed")
            await expect(row).not_to_contain_text("1 branch is open")
            for statuses, truncated, text in [
                (["active", "rolled_back"], False, "1 active, 1 rolled back"),
                ([], True, "unavailable or incomplete"),
                (["committed", "fenced"], True, "(incomplete)"),
                ([], False, "No branch handles are attached here"),
            ]:
                summary = WorkspaceBranchLifecycleSummary(
                    attached_count=len(statuses), statuses=statuses, truncated=truncated
                )
                workspace.branch_lifecycle_summary = lambda summary=summary: summary
                await page.reload(wait_until="networkidle")
                await expect(row).to_contain_text(text)

    asyncio.run(exercise())
