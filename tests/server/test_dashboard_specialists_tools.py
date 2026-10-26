"""Real-server acceptance for Specialists and Tools production pages."""

import asyncio
import os

import pytest
from playwright.async_api import expect
from tests.core.test_cloud_inventory import _Tool
from tests.server.test_dashboard_session_search import search_browser

from cayu import AgentSpec, CayuApp, ToolSpec
from cayu.context.thinking import ThinkingConfig
from cayu.evals.testing import ScriptedModelProvider

pytestmark = pytest.mark.skipif(
    os.environ.get("CAYU_AGENT_PAGES_LIVE") != "1", reason="Opt-in browser acceptance."
)


def _inventory(**options):
    app = CayuApp(enable_logging=False, **options)
    app.register_provider(ScriptedModelProvider([]), default=True)
    for name, description in [
        ("writer", "Writes release notes"),
        ("auditor", "Checks release notes"),
    ]:
        tool = _Tool("query_ledger")
        tool.spec = ToolSpec(
            name="query_ledger",
            description=description,
            input_schema={"type": "object", "properties": {name: {"type": "string"}}},
        )
        app.register_agent(
            AgentSpec(
                name=name,
                model="scripted",
                system_prompt="Private prompt",
                provider_options={"temperature": 0},
                thinking=ThinkingConfig(effort="high"),
                metadata={"cayu:description": description, "team": "release"},
            ),
            tools=[tool],
        )
    return app


@pytest.mark.parametrize("member", [False, True])
def test_agents_alias_preserves_search_and_specialist_configuration(member):
    async def run():
        async with search_browser(runtime=_inventory(), member=member) as (page, base, *_):
            await page.goto(base + "/cayu/agents?q=writer")
            await expect(page).to_have_url(base + "/cayu/specialists?q=writer")
            await expect(page.get_by_text("Writes release notes", exact=True).first).to_be_visible()
            await page.get_by_role("button", name="Inspect writer", exact=True).click()
            await expect(page.get_by_text("Provider options", exact=True)).to_be_visible()
            await expect(page.get_by_text("System prompt", exact=True)).to_be_visible()
            await expect(page.get_by_text("configured", exact=True)).to_be_visible()
            await expect(
                page.locator("article pre").filter(has_text="temperature")
            ).to_contain_text("0")
            await expect(page.locator("article pre").filter(has_text="effort")).to_contain_text(
                "high"
            )
            if member:
                await expect(
                    page.get_by_role("link", name="Start a session", exact=True)
                ).to_have_count(0)
            assert "Private prompt" not in await page.locator("main").inner_text()
            await expect(page.locator("main header")).to_have_count(1)

    asyncio.run(run())


@pytest.mark.parametrize("member", [False, True])
def test_tools_keep_same_name_declaration_variants(member):
    async def run():
        async with search_browser(runtime=_inventory(), member=member) as (page, base, *_):
            await page.goto(base + "/cayu/tools")
            await expect(page.get_by_role("heading", name="Tools", exact=True)).to_be_visible()
            await expect(page.get_by_role("button", name="Inspect query_ledger")).to_have_count(2)
            await page.get_by_role("button", name="Inspect query_ledger").first.click()
            await expect(page.get_by_text("Input schema", exact=True)).to_be_visible()
            await expect(page.get_by_text("Not called yet.", exact=True)).to_be_visible()

    asyncio.run(run())


async def _used_inventory():
    from tests.core.test_tool_completion import FinalTool, app_for, call
    from tests.core.test_tool_round_publication_failure_matrix import _TwoCallProvider

    from cayu import Message, RunRequest
    from cayu.providers import ModelStreamEvent
    from cayu.sessions import InMemorySessionStore

    app = app_for(
        InMemorySessionStore(),
        _TwoCallProvider(
            [
                *[call() for _ in range(23)],
                [ModelStreamEvent.text_delta("Done"), ModelStreamEvent.completed()],
            ]
        ),
        FinalTool(),
    )
    _ = [
        e
        async for e in app.run(
            RunRequest(
                session_id="tool-history",
                agent_name="support",
                messages=[Message.text("user", "Go")],
                max_steps=30,
            )
        )
    ]
    return app


def test_tool_usage_recent_pages_and_trace_link_bind_session_incarnation():
    from urllib.parse import parse_qs, urlparse

    from cayu import Message, RunRequest
    from cayu.sessions import SessionIdentity

    async def run():
        app = await _used_inventory()
        async with search_browser(runtime=app) as (page, base, *_):
            await page.goto(base + "/cayu/tools")
            await page.get_by_role("button", name="Inspect ask_customer", exact=True).click()
            links = page.get_by_role("link").filter(has_text="tool-history")
            await expect(links).to_have_count(20)
            await expect(page.locator("tbody tr").first.locator("td").nth(5)).to_have_text("23")
            await page.get_by_role("button", name="Load older calls", exact=True).click()
            await expect(links).to_have_count(23)
            href = await links.first.get_attribute("href")
            assert href
            query = parse_qs(urlparse(href).query)
            assert query["event_id"] and query["session_instance_id"]
            await links.first.click()
            await expect(page.get_by_role("tab", name="Trace", exact=True)).to_have_attribute(
                "aria-selected", "true"
            )
            await expect(page.get_by_label("Event ID", exact=True)).to_have_value(
                query["event_id"][0]
            )
            await page.goto(base + "/cayu/tools")
            await app.session_store.delete_session("tool-history")
            await app.session_store.create(
                RunRequest(
                    session_id="tool-history",
                    agent_name="support",
                    messages=[Message.text("user", "New incarnation")],
                ),
                identity=SessionIdentity(provider_name="scripted", model="scripted-model"),
            )
            await page.goto(base + href)
            await expect(
                page.get_by_text(
                    "This call no longer belongs to the retained session.", exact=False
                )
            ).to_be_visible()
            await expect(page.get_by_label("Event type", exact=True)).to_have_count(0)

    asyncio.run(run())


@pytest.mark.parametrize("state", ["pending", "error", "disabled"])
def test_tools_usage_state_never_becomes_false_zero(state):
    import re

    async def run():
        async with search_browser(runtime=_inventory()) as (page, base, *_):
            release = asyncio.Event()

            async def usage(route):
                if state == "pending":
                    await release.wait()
                    await route.continue_()
                else:
                    await route.fulfill(
                        status=501, json={"detail": "Usage unavailable from this store"}
                    )

            async def contract(route):
                response = await route.fetch()
                body = await response.json()
                body["capabilities"]["surfaces"]["tool_usage"]["read"] = {
                    "enabled": False,
                    "unavailable_reason": "unsupported",
                }
                await route.fulfill(response=response, json=body)

            if state == "disabled":
                await page.route("**/api/contract", contract)
            else:
                await page.route(re.compile(r"/api/tools/usage$"), usage)
            try:
                await page.goto(base + "/cayu/tools")
                await expect(page.get_by_role("button", name="Inspect query_ledger")).to_have_count(
                    2
                )
                if state == "pending":
                    await expect(
                        page.get_by_role("status").filter(has_text="Loading usage")
                    ).to_be_visible()
                elif state == "error":
                    await expect(page.get_by_role("alert")).to_contain_text("Usage unavailable")
                else:
                    await expect(
                        page.get_by_text("Tool usage is unavailable", exact=False)
                    ).to_be_visible()
                    forbidden_reads = []
                    page.on(
                        "request",
                        lambda request: (
                            forbidden_reads.append(request.url)
                            if "/api/tools/" in request.url
                            else None
                        ),
                    )
                    await page.get_by_role("button", name="Inspect query_ledger").first.click()
                    await asyncio.sleep(5.5)
                    assert forbidden_reads == []
                await expect(
                    page.locator("tbody td").filter(has_text=re.compile(r"^0$"))
                ).to_have_count(0)
            finally:
                release.set()

    asyncio.run(run())


@pytest.mark.parametrize("route", ["specialists", "tools"])
def test_agent_pages_dense_widths_search_focus_and_long_content(route):
    from pathlib import Path

    async def run():
        app = _inventory()
        app.register_agent(
            AgentSpec(
                name="long-" + "x" * 120,
                model="scripted",
                metadata={
                    "cayu:description": ("unbrokentext" * 160)[:1024],
                    "details": {"nested": "value" * 200},
                },
            )
        )
        async with search_browser(runtime=app) as (page, base, *_):
            await page.goto(base + "/cayu/" + route)
            search = page.get_by_role("textbox", name="Search " + route, exact=True)
            await expect(search).to_be_visible()
            await search.focus()
            await page.keyboard.type("writer" if route == "specialists" else "query")
            await expect(search).to_have_value("writer" if route == "specialists" else "query")
            await expect(search).to_be_focused()
            await search.fill("")
            for width in sorted(
                set([320, 390, 767, 768, 1023, 1024, 1279, 1440, *range(640, 1280, 16)])
            ):
                await page.set_viewport_size({"width": width, "height": 900})
                box = await search.bounding_box()
                assert box and box["width"] >= 170, (width, box)
                assert await page.evaluate("document.documentElement.scrollWidth<=innerWidth"), (
                    width
                )
            await page.set_viewport_size({"width": 320, "height": 900})
            inspect = page.get_by_role(
                "button",
                name="Inspect writer" if route == "specialists" else "Inspect query_ledger",
                exact=True,
            ).first
            assert await inspect.evaluate("e=>getComputedStyle(e).borderTopWidth") == "0px"
            await inspect.focus()
            await page.keyboard.press("Space")
            await expect(inspect).to_have_attribute("aria-expanded", "true")
            assert await page.evaluate("document.documentElement.scrollWidth<=innerWidth")
            for width in [320, 767, 1024, 1440]:
                await page.set_viewport_size({"width": width, "height": 900})
                for dark in [False, True]:
                    await page.evaluate(
                        "dark=>document.documentElement.classList.toggle('dark',dark)", dark
                    )
                    capture = Path(".agents/agent-page-captures")
                    capture.mkdir(exist_ok=True)
                    await page.screenshot(
                        path=str(capture / f"{route}-{width}-{'dark' if dark else 'light'}.png")
                    )

    asyncio.run(run())


@pytest.mark.parametrize("state", ["pending", "error", "empty"])
def test_specialist_inventory_distinguishes_loading_error_and_empty(state):
    async def run():
        async with search_browser(runtime=_inventory()) as (page, base, *_):
            release = asyncio.Event()

            async def inventory(route):
                if state == "pending":
                    await release.wait()
                    await route.continue_()
                elif state == "empty":
                    await route.fulfill(json={"agents": [], "total_count": 0})
                else:
                    await route.fulfill(
                        status=503,
                        json={"detail": "Specialists temporarily unavailable " + "x" * 300},
                    )

            await page.route("**/api/agents", inventory)
            try:
                await page.goto(base + "/cayu/specialists")
                if state == "pending":
                    await expect(
                        page.get_by_role("status").filter(has_text="Loading specialists")
                    ).to_be_visible()
                elif state == "empty":
                    await expect(page.get_by_text("No specialists", exact=True)).to_be_visible()
                else:
                    alert = page.get_by_role("alert")
                    await expect(alert).to_contain_text("temporarily unavailable", timeout=12000)
                    assert await alert.evaluate("e=>e.scrollWidth<=e.clientWidth")
            finally:
                release.set()

    asyncio.run(run())


def test_projected_identical_declarations_remain_counted_without_private_name_links():
    from cayu.vaults.redaction import SecretRedactor

    async def run():
        private_names = ["private_alpha", "private_beta"]
        app = CayuApp(secret_redactor=SecretRedactor(private_names), enable_logging=False)
        app.register_provider(ScriptedModelProvider([]), default=True)
        tools = []
        for name in private_names:
            tool = _Tool(name)
            tool.spec = ToolSpec(
                name=name, description="Shared public query", input_schema={"type": "object"}
            )
            tools.append(tool)
        app.register_agent(AgentSpec(name="specialist", model="scripted"), tools=tools)
        async with search_browser(runtime=app) as (page, base, *_):
            requests = []
            page.on("request", lambda request: requests.append(request.url))
            await page.goto(base + "/cayu/tools")
            inventory = await (await page.request.get(base + "/api/agents")).json()
            declarations = inventory["agents"][0]["tools"]
            assert len(declarations) == 2
            name = declarations[0]["name"]
            assert declarations[1]["name"] == name
            await page.get_by_role("button", name=f"Inspect {name}", exact=True).click()
            await expect(
                page.get_by_text("2 published declarations with this configuration.", exact=True)
            ).to_be_visible()
            await expect(page.get_by_text("Not called yet.", exact=True)).to_be_visible()
            body = await page.locator("main").inner_text()
            assert all(secret not in body for secret in private_names)
            assert all(secret not in url for secret in private_names for url in requests)

    asyncio.run(run())


def test_provider_hosted_outcomes_keep_unknown_distinct_from_failures():
    from tests.core.test_tool_round_publication_failure_matrix import _TwoCallProvider

    from cayu import Message, RunRequest
    from cayu.providers import ModelStreamEvent
    from cayu.providers.hosted import OpenAIWebSearch

    class HostedProvider(_TwoCallProvider):
        def preflight_hosted_tools(self, **kwargs):
            pass

    async def run():
        provider = HostedProvider(
            [
                [
                    *[
                        ModelStreamEvent.hosted_tool_call(
                            {
                                "tool_type": "web_search",
                                "call_id": f"hosted-{index}",
                                "status": status,
                            }
                        )
                        for index, status in enumerate(
                            ["in_progress", "completed", "failed", "incomplete", "outcome_unknown"]
                        )
                    ],
                    ModelStreamEvent.text_delta("Done"),
                    ModelStreamEvent.completed(),
                ]
            ]
        )
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="specialist", model="scripted"), hosted_tools=[OpenAIWebSearch()]
        )
        _ = [
            e
            async for e in app.run(
                RunRequest(
                    session_id="hosted-session",
                    agent_name="specialist",
                    messages=[Message.text("user", "Go")],
                )
            )
        ]
        async with search_browser(runtime=app, member=True) as (page, base, *_):
            await page.goto(base + "/cayu/tools")
            inventory = await (await page.request.get(base + "/api/agents")).json()
            name = inventory["agents"][0]["tools"][0]["name"]
            await page.get_by_role("button", name=f"Inspect {name}", exact=True).click()
            row = page.locator("tbody tr").first
            await expect(row.locator("td").nth(1)).to_contain_text("Provider")
            await expect(row.locator("td").nth(5)).to_have_text("4")
            await expect(row.locator("td").nth(6)).to_have_text("1")
            links = page.get_by_role("link").filter(has_text="hosted-session")
            await expect(links).to_have_count(4)
            await expect(links.filter(has_text="outcome unknown")).to_have_count(1)
            await expect(links.filter(has_text="incomplete")).to_have_count(1)

    asyncio.run(run())


def test_specialist_activity_and_tasks_use_bounded_specialist_queries_beyond_global_page():
    from cayu import Message, RunRequest
    from cayu.sessions import SessionIdentity
    from cayu.tasks import InMemoryTaskStore, TaskCreate

    async def run():
        tasks = InMemoryTaskStore()
        app = _inventory(task_store=tasks)
        for index in range(206):
            await app.session_store.create(
                RunRequest(
                    session_id="writer-history" if index == 0 else f"newer-{index}",
                    agent_name="writer" if index == 0 else "auditor",
                    title="Writer history" if index == 0 else "Other activity",
                    messages=[Message.text("user", "Go")],
                ),
                identity=SessionIdentity(provider_name="scripted", model="scripted"),
            )
        await tasks.create_task(
            TaskCreate(
                task_id="assigned-task",
                type="report",
                assigned_agent_name="writer",
                title="Assigned report",
                input={},
            )
        )
        async with search_browser(runtime=app, member=True) as (page, base, queries, *_):
            await page.goto(base + "/cayu/agents?q=writer")
            await page.get_by_role("button", name="Inspect writer", exact=True).click()
            await expect(
                page.get_by_role("link", name="Writer history", exact=True)
            ).to_be_visible()
            await expect(
                page.get_by_role("link", name="Assigned report", exact=True)
            ).to_be_visible()
            assert any(q.get("agent_name") == "writer" and q.get("limit") == "5" for q in queries)
            assert all(q.get("limit") == "5" for q in queries)
            await page.goto(base + "/cayu/agents?q=Checks%20release%20notes")
            await expect(
                page.get_by_role("textbox", name="Search specialists", exact=True)
            ).to_have_value("Checks release notes")
            await expect(
                page.get_by_role("button", name="Inspect auditor", exact=True)
            ).to_be_visible()
            await expect(
                page.get_by_role("button", name="Inspect writer", exact=True)
            ).to_have_count(0)
            await page.go_back()
            await expect(
                page.get_by_role("textbox", name="Search specialists", exact=True)
            ).to_have_value("writer")

    asyncio.run(run())


def test_mounted_tool_usage_and_history_refresh_with_bounded_reads():
    from tests.core.test_tool_completion import FinalTool, app_for, call
    from tests.core.test_tool_round_publication_failure_matrix import _TwoCallProvider

    from cayu import Message, RunRequest
    from cayu.providers import ModelStreamEvent
    from cayu.sessions import InMemorySessionStore

    async def run():
        done = [ModelStreamEvent.text_delta("Done"), ModelStreamEvent.completed()]
        app = app_for(
            InMemorySessionStore(),
            _TwoCallProvider([*[call() for _ in range(23)], done, call(), done]),
            FinalTool(),
        )

        async def execute(name):
            _ = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id=name,
                        agent_name="support",
                        messages=[Message.text("user", "Go")],
                        max_steps=30,
                    )
                )
            ]

        await execute("previous-calls")
        async with search_browser(runtime=app) as (page, base, *_):
            requests = []
            page.on(
                "request",
                lambda request: (
                    requests.append(request.url) if "/api/tools/" in request.url else None
                ),
            )
            await page.goto(base + "/cayu/tools")
            await page.get_by_role("button", name="Inspect ask_customer", exact=True).click()
            await expect(page.get_by_role("link").filter(has_text="previous-calls")).to_have_count(
                20
            )
            await page.get_by_role("button", name="Load older calls", exact=True).click()
            await expect(page.get_by_role("link").filter(has_text="previous-calls")).to_have_count(
                23
            )
            requests.clear()
            await execute("fresh-call")
            await expect(page.locator("tbody tr").first.locator("td").nth(5)).to_have_text(
                "24", timeout=12000
            )
            await expect(page.get_by_role("link").filter(has_text="fresh-call")).to_have_count(
                1, timeout=12000
            )
            await expect(page.get_by_role("link").filter(has_text="previous-calls")).to_have_count(
                23
            )
            assert not any("cursor=" in url for url in requests), requests
            assert len(requests) <= 8, requests
            await page.get_by_role("button", name="Start from latest calls", exact=True).click()
            await expect(page.get_by_role("link").filter(has_text="previous-calls")).to_have_count(
                19
            )
            await expect(page.get_by_role("link").filter(has_text="fresh-call")).to_have_count(1)
            await page.goto(base + "/cayu/")
            requests.clear()
            await asyncio.sleep(5.5)
            assert requests == []

    asyncio.run(run())


def test_mounted_specialist_activity_refreshes_after_new_work():
    from cayu import Message, RunRequest
    from cayu.sessions import SessionIdentity
    from cayu.tasks import InMemoryTaskStore, TaskCreate

    async def run():
        tasks = InMemoryTaskStore()
        app = _inventory(task_store=tasks)
        async with search_browser(runtime=app) as (page, base, *_):
            await page.goto(base + "/cayu/specialists?q=writer")
            await page.get_by_role("button", name="Inspect writer", exact=True).click()
            await expect(page.get_by_text("No recent sessions found", exact=True)).to_be_visible()
            await app.session_store.create(
                RunRequest(
                    session_id="arriving-session",
                    agent_name="writer",
                    title="Arriving activity",
                    messages=[Message.text("user", "Go")],
                ),
                identity=SessionIdentity(provider_name="scripted", model="scripted"),
            )
            await tasks.create_task(
                TaskCreate(
                    task_id="arriving-task",
                    type="report",
                    assigned_agent_name="writer",
                    title="Arriving report",
                    input={},
                )
            )
            await expect(
                page.get_by_role("link", name="Arriving activity", exact=True)
            ).to_be_visible(timeout=12000)
            await expect(
                page.get_by_role("link", name="Arriving report", exact=True)
            ).to_be_visible(timeout=12000)

    asyncio.run(run())
