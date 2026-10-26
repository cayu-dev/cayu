"""Registered tool declarations reach Session prose without borrowing another owner."""

from __future__ import annotations

import asyncio
import os

import pytest
from tests.core.test_tool_display import DisplayTool
from tests.server.test_dashboard_prototype_links import prototype_url
from tests.server.test_dashboard_session_search import search_browser

from cayu import AgentSpec, CayuApp, Message, RunRequest, ToolSpec
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.vaults import SecretRedactor

__all__ = ["prototype_url"]

pytestmark = pytest.mark.skipif(
    os.environ.get("CAYU_SESSION_DETAIL_LIVE") != "1",
    reason="Opt-in packaged Chromium tool display acceptance.",
)


@pytest.mark.parametrize(
    "mode",
    [
        "native",
        "native-desktop",
        "agent-collision",
        "tool-collision",
        "hidden-agent",
        "hidden-tool",
        "disabled",
        "fallback",
        "inventory-error",
        "inventory-delayed",
    ],
)
def test_session_labels_use_only_unambiguous_enabled_declarations(mode):
    from playwright.async_api import expect

    async def run():
        hidden_agent = mode in {"agent-collision", "hidden-agent"}
        hidden_tool = mode in {"tool-collision", "hidden-tool"}
        secrets = ["private_owner_one", "private_owner_two"] if hidden_agent else []
        secrets += ["private_tool_one", "private_tool_two"] if hidden_tool else []
        runtime = CayuApp(enable_logging=False)
        owner = "private_owner_one" if hidden_agent else "reader"
        name = (
            "private_tool_one"
            if hidden_tool
            else (
                "constructor"
                if mode.startswith("inventory-")
                else "read_file"
                if mode == "fallback"
                else "inspect_entries"
            )
        )
        calls = [
            ModelStreamEvent.tool_call(name=name, arguments={}, id=f"call-{index}")
            for index in range(2)
        ]
        runtime.register_provider(
            ScriptedModelProvider(
                [
                    [*calls, ModelStreamEvent.completed({"finish_reason": "tool_calls"})],
                    [ModelStreamEvent.text_delta("Finished"), ModelStreamEvent.completed()],
                ]
            ),
            default=True,
        )
        declaration = ToolSpec(
            name=name,
            display=None
            if mode == "fallback"
            else {"verb": "Read", "noun": "entry", "plural": "entries"},
        )
        tools = [DisplayTool(declaration)]
        if mode == "tool-collision":
            tools.append(
                DisplayTool(ToolSpec(name="private_tool_two", display={"verb": "Borrowed"}))
            )

        # Both read owners register identical declarations in the same order.
        def register(app):
            app.register_agent(
                AgentSpec(
                    name="private_owner_two" if mode == "agent-collision" else "other",
                    model="scripted",
                ),
                tools=[DisplayTool(ToolSpec(name=name, display={"verb": "Borrowed"}))],
            )
            app.register_agent(AgentSpec(name=owner, model="scripted"), tools=tools)

        register(runtime)
        _ = [
            event
            async for event in runtime.run(
                RunRequest(
                    session_id="label-session",
                    agent_name=owner,
                    messages=[Message.text("user", "Inspect these entries")],
                )
            )
        ]
        assert (await runtime.session_store.load("label-session")).status == "completed"
        original_runtime = runtime
        if secrets:
            # The session predates discovery of a secret. A new read owner sees
            # the same retained records through the current redaction boundary.
            runtime = CayuApp(
                enable_logging=False,
                session_store=original_runtime.session_store,
                secret_redactor=SecretRedactor(secrets),
            )
            runtime.register_provider(ScriptedModelProvider([]), default=True)
            register(runtime)
        async with search_browser(runtime=runtime, member=mode == "native") as (
            page,
            base,
            _,
            _,
            _,
        ):
            failures = []
            page.on("pageerror", lambda error: failures.append(str(error)))
            inventory = await page.request.get(f"{base}/api/agents")
            assert inventory.ok
            inventory_body = await inventory.json()
            assert any(
                tool["display"] for agent in inventory_body["agents"] for tool in agent["tools"]
            )
            if mode == "disabled":

                async def disable(route):
                    response = await route.fetch()
                    body = await response.json()
                    body["capabilities"]["surfaces"]["tool_display"]["read"] = {
                        "enabled": False,
                        "unavailable_reason": "unsupported",
                    }
                    await route.fulfill(response=response, json=body)

                await page.route("**/api/contract", disable)
            release = asyncio.Event()
            if mode == "inventory-error":
                await page.route(
                    "**/api/agents",
                    lambda route: route.fulfill(
                        status=503, json={"detail": "Inventory unavailable"}
                    ),
                )
            elif mode == "inventory-delayed":

                async def delayed(route):
                    await release.wait()
                    await route.continue_()

                await page.route("**/api/agents", delayed)
            await page.goto(f"{base}/cayu/sessions/label-session")
            if mode == "inventory-delayed":
                try:
                    await expect(
                        page.get_by_role("button", name="Used constructor 2 times", exact=False)
                    ).to_be_visible()
                finally:
                    release.set()
            await expect(page.get_by_text("Finished", exact=True)).to_be_visible()
            public_name = "[REDACTED_SECRET]" if hidden_tool else name
            labelled = mode in {"native", "native-desktop", "inventory-delayed"}
            summary = "Read 2 entries" if labelled else f"Used {public_name} 2 times"
            await page.get_by_role("button", name=summary, exact=False).click()
            row = page.locator('[id^="transcript-tool-"]').first
            await expect(row.get_by_role("button").first).to_have_text(
                "Read" if labelled else public_name
            )
            await row.get_by_role("button").first.click()
            if labelled:
                trace_response = await page.request.get(
                    f"{base}/api/sessions/label-session/trace?limit=1000"
                )
                spans = [s for s in (await trace_response.json())["spans"] if s["kind"] == "tool"]
                assert len(spans) == 2
                position = spans[0]["transcript_links"][0]
                assert (
                    await row.get_attribute("id")
                    == f"transcript-tool-{position['message_index']}-{position['part_index']}"
                )
                await row.get_by_role("button", name="View in trace", exact=True).click()
                tree = page.get_by_role("tree", name="Session trace")
                await expect(tree.get_by_role("treeitem").filter(has_text="Read")).to_have_count(2)
                detail = page.get_by_role("complementary", name="Span detail")
                await detail.get_by_role("button", name="Show in conversation", exact=True).click()
                await expect(row.get_by_role("button").first).to_have_attribute(
                    "aria-expanded", "true"
                )
                assert await page.evaluate(
                    "document.documentElement.scrollWidth <= window.innerWidth"
                )
            if mode.startswith("native") and os.environ.get("CAYU_TOOL_DISPLAY_SCREENSHOTS"):
                from pathlib import Path

                output = Path(os.environ["CAYU_TOOL_DISPLAY_SCREENSHOTS"])
                output.mkdir(parents=True, exist_ok=True)
                await page.screenshot(path=str(output / f"{mode}.png"), full_page=True)
            assert "Borrowed" not in await page.locator("body").inner_text()
            assert not failures
        await runtime.aclose()
        if original_runtime is not runtime:
            await original_runtime.aclose()

    asyncio.run(run())


async def _presentation_runtime(
    names_and_labels, *, approval=False, failed=False, target="src/entries.json"
):
    from cayu import ToolResult
    from cayu.tools.policy import AlwaysRequireApprovalToolPolicy

    class PresentationTool(DisplayTool):
        async def run(self, ctx, args):
            return ToolResult(content="Recorded result", is_error=failed)

    runtime = CayuApp(enable_logging=False)
    runtime.register_provider(
        ScriptedModelProvider(
            [
                [
                    *[
                        ModelStreamEvent.tool_call(
                            name=name, arguments={"path": target}, id=f"c{index}"
                        )
                        for index, (name, _) in enumerate(names_and_labels)
                    ],
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [ModelStreamEvent.text_delta("Finished"), ModelStreamEvent.completed()],
            ]
        ),
        default=True,
    )
    tools = [
        PresentationTool(
            ToolSpec(
                name=name,
                display={"verb": verb} if verb else None,
                input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
            )
        )
        for name, verb in names_and_labels
    ]
    runtime.register_agent(
        AgentSpec(name="reader", model="scripted"),
        tools=tools,
        **({"tool_policy": AlwaysRequireApprovalToolPolicy()} if approval else {}),
    )
    _ = [
        event
        async for event in runtime.run(
            RunRequest(
                session_id="presentation",
                agent_name="reader",
                messages=[Message.text("user", "Inspect")],
            )
        )
    ]
    return runtime


LONG_TARGET = "src/" + "very-long-command-or-path/" * 100
LONG_VERB = (
    "Retrieved and validated the customer transaction history and reconciled every historical entry"
)


@pytest.mark.parametrize(
    "verb,target",
    [
        (LONG_VERB, "src/entries.json"),
        ("Validated" * 30, "src/entries.json"),
        ("Ran", LONG_TARGET),
        (LONG_VERB, LONG_TARGET),
        ("Validated" * 30, LONG_TARGET),
    ],
    ids=[
        "long-words",
        "long-unbroken",
        "short-long-target",
        "words-long-target",
        "unbroken-long-target",
    ],
)
def test_native_labels_keep_priority_and_fit_conversation_and_trace(verb, target):
    from playwright.async_api import expect

    async def run():
        runtime = await _presentation_runtime(
            [("inspect", verb), ("short_read", "Read")], failed=True, target=target
        )
        async with search_browser(runtime=runtime) as (page, base, _, _, _):
            for width in [390, 800, 1440]:
                await page.set_viewport_size({"width": width, "height": 900})
                await page.goto(f"{base}/cayu/sessions/presentation")
                await expect(page.get_by_text("Finished", exact=True)).to_be_visible()
                await page.get_by_role("button", name=verb, exact=False).click()
                row = page.locator('[id^="transcript-tool-"]').first
                button = row.get_by_role("button").first
                label = button.locator(":scope > span").first
                await expect(label).to_have_text(verb)
                # Check each real label/argument/status/control, not document width.
                bounds = await button.evaluate("""node => {
                    const row = node.getBoundingClientRect();
                    return [...node.children].map(child => {
                        const box = child.getBoundingClientRect();
                        return {inside: box.left >= row.left - 1 && box.right <= row.right + 1, width: box.width};
                    });
                }""")
                assert all(item["inside"] for item in bounds), bounds
                assert bounds[1]["width"] >= 50, bounds  # argument preview remains usable
                assert await label.evaluate("""node => {
                    const context = document.createElement('canvas').getContext('2d');
                    context.font = getComputedStyle(node).font;
                    return node.clientWidth + 1 >= Math.min(80, context.measureText(node.textContent).width);
                }"""), bounds
                await expect(button.get_by_text("failed", exact=True)).to_be_visible()
                await button.click()
                await row.get_by_role("button", name="View in trace", exact=True).click()
                tree_row = (
                    page.get_by_role("tree", name="Session trace")
                    .get_by_role("treeitem")
                    .filter(has_text=verb)
                )
                await expect(tree_row).to_have_count(1)
                column = tree_row.locator(":scope > span").first
                trace_label = column.get_by_text(verb, exact=True)
                geometry = await trace_label.evaluate("""node => {
                    const label = node.getBoundingClientRect(), column = node.parentElement.getBoundingClientRect();
                    const timing = node.parentElement.nextElementSibling.getBoundingClientRect();
                    return {inside: label.left >= column.left && label.right <= column.right + 1, separate: label.right <= timing.left};
                }""")
                assert geometry["inside"] and geometry["separate"], geometry
                assert await trace_label.evaluate("""node => {
                    const context = document.createElement('canvas').getContext('2d');
                    context.font = getComputedStyle(node).font;
                    return node.clientWidth + 1 >= Math.min(60, context.measureText(node.textContent).width);
                }"""), await trace_label.bounding_box()
                # Keyboard selection opens the complete label, including unbroken text.
                await tree_row.focus()
                await page.keyboard.press("Enter")
                detail = page.get_by_role("complementary", name="Span detail")
                await expect(detail.get_by_text(verb, exact=True)).to_be_visible()
                assert await detail.evaluate("node => node.scrollWidth <= node.clientWidth"), width
                await detail.get_by_role("button", name="Show in conversation", exact=True).click()
                await expect(button).to_have_attribute("aria-expanded", "true")
                short = page.locator('[id^="transcript-tool-"]').nth(1).get_by_role("button").first
                await expect(short.get_by_text("Read", exact=True)).to_be_visible()
                assert (await short.bounding_box())["height"] < 40
                if width == 390 and os.environ.get("CAYU_TOOL_DISPLAY_SCREENSHOTS"):
                    from pathlib import Path

                    output = Path(os.environ["CAYU_TOOL_DISPLAY_SCREENSHOTS"])
                    output.mkdir(parents=True, exist_ok=True)
                    kind = "words" if " " in verb else "unbroken"
                    if target == LONG_TARGET:
                        kind += "-long-target"
                    await page.screenshot(path=str(output / f"long-{kind}.png"), full_page=True)
        await runtime.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("approval", [False, True])
@pytest.mark.parametrize("declared", [False, True])
def test_trace_detail_special_tool_names_require_own_display(approval, declared):
    from playwright.async_api import expect

    async def run():
        names = ["constructor", "toString", "__proto__"]
        runtime = await _presentation_runtime(
            [(name, "Inspected" if declared else None) for name in names], approval=approval
        )
        async with search_browser(runtime=runtime) as (page, base, _, _, _):
            if not declared:
                await page.route(
                    "**/api/agents",
                    lambda route: route.fulfill(
                        status=503, json={"detail": "Inventory unavailable"}
                    ),
                )
            await page.goto(f"{base}/cayu/sessions/presentation?view=trace")
            response = await page.request.get(f"{base}/api/sessions/presentation/trace?limit=1000")
            trace = await response.json()
            spans = [s for s in trace["spans"] if s["kind"] in {"tool", "wait"}]
            events = {event["id"]: event for event in trace["events"]}
            assert spans
            assert any(s["kind"] == ("wait" if approval else "tool") for s in spans)
            for span in spans:
                await page.goto(f"{base}/cayu/sessions/presentation?view=trace&span={span['id']}")
                detail = page.get_by_role("complementary", name="Span detail")
                fact = (
                    detail.locator("div")
                    .filter(has=page.locator("dt").get_by_text("Tool", exact=True))
                    .last.locator("dd")
                )
                await expect(fact).to_be_visible()
                first = events[span["event_ids"][0]]
                has_tool = first.get("tool_name") in names
                if declared and has_tool:
                    await expect(fact).to_contain_text("(Inspected)")
                else:
                    await expect(fact).not_to_contain_text("(Inspected)")
                text = await fact.inner_text()
                assert "undefined" not in text
                if has_tool:
                    assert first["tool_name"] in text
        await runtime.aclose()

    asyncio.run(run())


def test_prototype_model_and_command_names_keep_priority_over_long_details(prototype_url):
    from playwright.sync_api import expect, sync_playwright

    with sync_playwright() as driver:
        browser = driver.chromium.launch(headless=True)
        try:
            page = browser.new_page(has_touch=True)
            url = f"{prototype_url}#/sessions/close-00a4224687bc?view=trace"
            page.goto(url)
            expect(page.get_by_role("tree", name="Session trace")).to_be_visible()
            page.evaluate(
                """async target => {
                const {SEEDS} = await import('/src/next/mock/seeds/index.ts');
                const session = SEEDS[0].sessions.find(s => s.session.id === 'close-00a4224687bc');
                for (const event of session.events) {
                    if (event.type.startsWith('model.')) event.payload.model = target;
                }
                const index = session.events.findIndex(e => e.type === 'tool.call.started');
                if (index < 0) throw new Error('Fixture needs a tool call');
                const tool = session.events[index];
                session.events.splice(index + 1, 0, ...['started', 'completed'].map((phase, index) => ({
                    ...structuredClone(tool), id: `label-command-${phase}`, type: `runner.exec.${phase}`,
                    payload: {...tool.payload, execution_id: 'label-command', command: target, exit_code: index ? 0 : null}
                })));
                session.events.forEach((event, index) => { event.sequence = index + 1; });
            }""",
                LONG_TARGET,
            )
            page.get_by_role("button", name="Reset data", exact=True).click()
            for width in [390, 800, 1440]:
                page.set_viewport_size({"width": width, "height": 900})
                tree = page.get_by_role("tree", name="Session trace")
                for name in ["Model step 1", "Command"]:
                    row = (
                        tree.get_by_role("treeitem")
                        .filter(has=page.get_by_text(name, exact=True))
                        .first
                    )
                    label = row.locator(":scope > span").first.get_by_text(name, exact=True)
                    expect(label).to_be_visible()
                    assert label.evaluate("node => node.clientWidth >= node.scrollWidth"), name
                    row.tap()
                    detail = page.get_by_role("complementary", name="Span detail")
                    expect(detail.locator("p.break-words").first).to_have_text(name)
                    row.focus()
                    page.keyboard.press("Enter")
                    expect(detail.locator("p.break-words").first).to_have_text(name)
        finally:
            browser.close()
