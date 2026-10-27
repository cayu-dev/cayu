"""Home/Inbox acceptance through a real server with signed viewer identities."""

import asyncio
import importlib
import os
import re
from pathlib import Path

import pytest

pytest.importorskip("playwright.async_api")
import socket
from contextlib import asynccontextmanager, suppress
from urllib.parse import parse_qs, urlsplit

import uvicorn
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import FastAPI
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import async_playwright, expect
from tests.server.test_proxy_identity import FakeJwks, _as, _auth, _token
from tests.server.test_server_inbox import _app, _knowledge_app

from cayu.approvals import HumanReviewDisclosure, HumanReviewField, HumanReviewPolicy
from cayu.messages import Message
from cayu.server import ServerConfig, create_server
from cayu.server.config import DashboardConfig
from cayu.sessions.requests import RunRequest
from cayu.tasks import InMemoryTaskStore, TaskCreate

pytestmark = pytest.mark.skipif(
    os.environ.get("CAYU_DASHBOARD_HOME_INBOX_LIVE") != "1",
    reason="Opt-in packaged Home/Inbox Chromium acceptance.",
)


@asynccontextmanager
async def _home_browser(
    *,
    role="admin",
    nested=False,
    mobile=False,
    decisions=False,
    protected=False,
    knowledge=False,
    limited=False,
    tasks=False,
    worker_agent=None,
    active_peer=False,
    usage=False,
    priced=False,
    failed=False,
    recovery=False,
    hidden_count=11,
    capture_runtime=None,
    long_knowledge=False,
    seeded_runtime=None,
    dashboard_config=None,
):
    key = ec.generate_private_key(ec.SECP256R1())
    auth = _auth(FakeJwks(key))

    class Review(HumanReviewPolicy):
        version = "home-inbox-test"

        @property
        def binding_key(self):
            return b"r" * 32

        def authorize(self, context, **kwargs):
            return context.purpose == "inbox"

        def project(self, context, source):
            return HumanReviewDisclosure(
                status="permitted",
                sensitive_content="application_attested",
                fields=(HumanReviewField(label="Safe note", text="Review this exact proposal"),),
            )

    task_store = InMemoryTaskStore()
    runtime = _app(
        task_store=task_store, **({"human_review_policy": Review()} if protected else {})
    )
    if tasks:
        await runtime.create_task(
            TaskCreate(task_id="needs-review", type="review", title="Review older invoice")
        )
        await task_store.block_task("needs-review", reason="Needs operator review")
        for index in range(12):
            await runtime.create_task(
                TaskCreate(task_id=f"recent-{index}", type="work", title=f"Recent task {index}")
            )
    if failed:
        from cayu.agents import AgentSpec
        from cayu.evals import ScriptedModelProvider

        def fail(request):
            raise RuntimeError("Expected browser fixture failure")

        runtime.register_provider(
            ScriptedModelProvider(name="failure", response_factory=fail), default=True
        )
        runtime.register_agent(AgentSpec(name="failing", model="broken"))
        async for _event in runtime.run(
            RunRequest(
                agent_name="failing",
                session_id="failed-session",
                messages=[Message.text("user", "go")],
            )
        ):
            pass
    if recovery:
        from tests.server.test_server_inbox import (
            _CountingTool,
            _crashed_user_input_resume_events,
            _ScriptedProvider,
            private_events_for_public_events,
        )

        from cayu import CayuApp
        from cayu.agents import AgentSpec
        from cayu.approvals.user_input import UserInputResponse
        from cayu.events import EventType
        from cayu.tools.user_input import UserInputTool

        runtime = CayuApp(enable_logging=False)
        runtime.register_provider(
            _ScriptedProvider(
                [("sibling", "count", {}), ("question", "ask_user", {"question": "Confirm"})]
            ),
            default=True,
        )
        runtime.register_agent(
            AgentSpec(name="asker", model="fake-model"), tools=[_CountingTool(), UserInputTool()]
        )
        paused = [
            event
            async for event in runtime.run(
                RunRequest(
                    agent_name="asker", session_id="crashed", messages=[Message.text("user", "go")]
                )
            )
        ]
        awaiting = next(
            event for event in paused if event.type is EventType.SESSION_AWAITING_USER_INPUT
        )
        await runtime.session_store.append_events(
            "crashed",
            _crashed_user_input_resume_events(
                await private_events_for_public_events(runtime.session_store, paused),
                session_id="crashed",
                tool_call_id="sibling",
            ),
        )
        stopped = [
            event
            async for event in runtime.resolve_user_input(
                UserInputResponse(
                    session_id="crashed", input_id=awaiting.payload["input_id"], answer="staging"
                )
            )
        ]
        assert any(event.payload.get("manual_recovery_required") for event in stopped)
    if usage:
        from tests.core.test_cloud_reporting import _app as usage_app
        from tests.core.test_cloud_reporting import _start

        from cayu.sessions import ForkSessionRequest
        from cayu.sessions._run_identity import eval_trial_scope

        runtime = usage_app()
        await _start(runtime, "normal", "writer")
        async with eval_trial_scope():
            await _start(runtime, "eval-trial", "writer")
        async for _event in runtime.fork_session(
            ForkSessionRequest(source_session_id="normal", session_id="forked")
        ):
            pass
    if worker_agent:
        from uuid import uuid4

        from cayu.tasks.worker import TaskHandlerOutcome, run_task_worker

        async def pause(app, task, worker_id):
            async for _event in app.run(
                RunRequest(
                    agent_name=worker_agent,
                    session_id="worker-session",
                    task_id=task.id,
                    task_worker_id=worker_id,
                    task_lease_expires_at=task.lease_expires_at,
                    messages=[Message.text("user", "go")],
                )
            ):
                pass
            return TaskHandlerOutcome.SESSION_INTERRUPTED

        await task_store.create_task(TaskCreate(task_id="worker-task", type="job"))
        await run_task_worker(
            runtime, task_store, pause, worker_id="original", max_tasks=1, reclaim=False
        )
        if active_peer:
            await task_store.claim_interrupted_task_continuation(
                "active-peer", handoff_id=str(uuid4()), task_id="worker-task"
            )
    if knowledge:
        runtime, knowledge_store = _knowledge_app()
        if long_knowledge:
            from cayu.storage import KnowledgeEntry, KnowledgeStatus

            await knowledge_store.create_entry(
                KnowledgeEntry(
                    id="long-proposal",
                    title="Long proposal",
                    text="Visible preview. " * 50 + "FINAL REVIEW CONDITION",
                    status=KnowledgeStatus.PENDING,
                )
            )
    if decisions:
        for name, agent in [("approval", "approver"), ("question", "asker")]:
            async for _event in runtime.run(
                RunRequest(agent_name=agent, session_id=name, messages=[Message.text("user", "go")])
            ):
                pass
    if limited:
        for name in ["visible", *[f"hidden-{index}" for index in range(hidden_count)]]:
            async for _event in runtime.run(
                RunRequest(
                    agent_name="approver", session_id=name, messages=[Message.text("user", "go")]
                )
            ):
                pass
        original = runtime.session_store.query_pending_actions

        async def bounded(query, **kwargs):
            return await original(query.model_copy(update={"limit": 1}), **kwargs)

        runtime.session_store.query_pending_actions = bounded

    async def access(person, item):
        return item.session_id == "visible"

    runtime_config = {"humanReviewPurpose": "inbox"} if protected else {}
    if priced:
        from cayu.budgets.pricing import default_price_book

        runtime_config["priceBook"] = default_price_book()
    if seeded_runtime is not None:
        runtime = seeded_runtime
    if dashboard_config is not None:
        runtime_config.update(dashboard_config)
    if capture_runtime:
        capture_runtime(runtime)
    app = create_server(
        runtime,
        inbox_access=access if limited else None,
        config=ServerConfig.protected(
            auth,
            deployment_name="Shell acceptance",
            dashboard=DashboardConfig(
                directory=os.environ.get("CAYU_DASHBOARD_SHELL_DIRECTORY"),
                runtime_config=runtime_config,
            ),
        ),
    )
    prefix = "/product" if nested else ""
    if nested:
        parent = FastAPI()
        parent.mount(prefix, app)
        app = parent
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(128)
        listener.setblocking(False)
        server = uvicorn.Server(
            uvicorn.Config(app, lifespan="on", log_level="error", timeout_graceful_shutdown=3)
        )
        task = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            async with asyncio.timeout(10):
                while not server.started:
                    if task.done():
                        await task
                    await asyncio.sleep(0.01)
            async with async_playwright() as playwright:
                browser = await playwright.chromium.launch(
                    executable_path=os.environ.get("CAYU_BROWSER_CONTROL_CHROMIUM"), headless=True
                )
                try:
                    context = await browser.new_context(
                        extra_http_headers=_as(_token(key, role=role)),
                        viewport={"width": 390 if mobile else 1440, "height": 900},
                        color_scheme="dark",
                    )
                    page = await context.new_page()
                    page.set_default_timeout(5_000)
                    requests = []
                    responses = []
                    page.on("request", lambda request: requests.append(urlsplit(request.url).path))
                    page.on("response", lambda response: responses.append(response))
                    base = f"http://127.0.0.1:{listener.getsockname()[1]}{prefix}"
                    yield page, base, requests, responses
                finally:
                    await browser.close()
        finally:
            server.should_exit = True
            try:
                await asyncio.wait_for(asyncio.shield(task), 10)
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)


def test_home_uses_real_daily_counts_and_shared_start_at_nested_base(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples"))
    example = importlib.import_module("dashboard_behavior_live")

    async def scenario():
        async with _home_browser(nested=True) as (page, base, requests, responses):
            await page.goto(base + "/cayu/?start=1", wait_until="networkidle")
            await expect(page.get_by_text("Last 24 hours:", exact=False)).to_be_visible()
            await expect(
                page.get_by_role("heading", name="Needs attention", exact=True)
            ).to_be_visible()
            await expect(
                page.get_by_role("heading", name="Recent sessions", exact=True)
            ).to_be_visible()
            await expect(page.get_by_role("tab", name="Run a specialist")).to_be_visible()
            await expect(
                page.get_by_text("No sessions need attention.", exact=True)
            ).to_be_visible()
            await expect(page.get_by_text("No recent sessions found.", exact=True)).to_be_visible()
            assert "/product/api/sessions/overview/counts" in requests
            await example._exercise_dashboard_home(page, base)
            await example._exercise_dashboard_inbox(page, base)
            await page.goto(base + "/cayu/run", wait_until="networkidle")
            await expect(page).to_have_url(re.compile(r"/cayu/\?start=1$"))
            await expect(page.get_by_label("Prompt", exact=True)).to_be_visible()
            assert not any(r.status in (401, 403) for r in responses)

    asyncio.run(scenario())


@pytest.mark.parametrize("decision", ["Approve", "Deny"])
def test_inbox_member_decides_without_arguments_and_counts_refresh(decision):
    async def scenario():
        async with _home_browser(role="member", decisions=True) as (
            page,
            base,
            requests,
            responses,
        ):
            await page.goto(base + "/cayu/pending-actions", wait_until="networkidle")
            await expect(page).to_have_url(base + "/cayu/inbox")
            await expect(
                page.get_by_role("button", name="Approve", exact=True).first
            ).to_be_visible()
            await expect(
                page.get_by_text("Arguments are withheld", exact=False).first
            ).to_be_visible()
            await expect(page.get_by_label("Changed arguments", exact=True)).to_have_count(0)
            assert "/api/inbox/counts" in requests
            await expect(page.get_by_label("Inbox total")).to_have_text("2")
            await page.get_by_text("Advanced filters", exact=True).click()
            await page.get_by_label("Inbox page size", exact=True).select_option("25")
            await expect(page.get_by_label("Inbox total")).to_have_text("2")
            await page.get_by_role("button", name=decision, exact=True).first.click()
            await expect(page.get_by_role("button", name="Approve", exact=True)).to_have_count(0)
            await expect(page.get_by_label("Inbox total")).to_have_text("1")
            await page.get_by_label("Your answer", exact=True).fill("Confirmed by the operator")
            await page.get_by_text("Decision options", exact=True).click()
            await page.get_by_label("Reason (optional)", exact=True).fill("Answer reviewed")
            await page.get_by_role("button", name="Send", exact=True).click()
            await expect(page.get_by_label("Your answer", exact=True)).to_have_count(0)
            await expect(page.get_by_label("Inbox total")).to_have_count(0)
            settlements = [
                await response.json()
                for response in responses
                if response.request.method == "POST"
                and urlsplit(response.url).path.endswith("/settle")
                and response.status == 200
            ]
            assert settlements[-1]["item"]["settlement"]["reason"] == "Answer reviewed"
            assert any(
                "/api/inbox/" in request and request.endswith("/settle") for request in requests
            )
            assert not any(r.status in (401, 403) for r in responses)
            await page.goto(base + "/cayu/?start=1", wait_until="networkidle")
            await expect(page.get_by_role("tab", name="Run a specialist")).to_have_count(0)
            assert "/api/run" not in requests

    asyncio.run(scenario())


def test_inbox_protected_review_uses_reference_for_member():
    async def scenario():
        async with _home_browser(role="member", decisions=True, protected=True) as (
            page,
            base,
            requests,
            responses,
        ):
            await page.goto(base + "/cayu/inbox?kind=approval", wait_until="networkidle")
            await expect(page.get_by_text("Protected human review", exact=True)).to_be_visible()
            await expect(page.get_by_role("button", name="Approve", exact=True)).to_have_count(0)
            await page.get_by_role("button", name="Refresh review").click()
            await expect(page.get_by_text("Review this exact proposal", exact=True)).to_be_visible()
            await expect(page.get_by_label("Decision reason", exact=True)).to_be_visible()
            await page.get_by_label("Decision reason", exact=True).fill("Checked by reviewer")
            await page.get_by_role("button", name="Approve", exact=True).click()
            await expect(page.get_by_role("button", name="Approve", exact=True)).to_have_count(0)
            assert any("human-review" in request for request in requests)
            assert any("/settle" in request for request in requests)
            assert not any(response.status >= 400 for response in responses)

    asyncio.run(scenario())


def test_inbox_member_keeps_and_drops_knowledge_with_source_fields():
    async def scenario():
        async with _home_browser(role="member", knowledge=True) as (
            page,
            base,
            requests,
            responses,
        ):
            await page.goto(base + "/cayu/inbox?kind=knowledge", wait_until="networkidle")
            await expect(page.get_by_role("button", name="Keep", exact=True)).to_have_count(2)
            for label in [
                "Namespace",
                "Labels",
                "Aspects",
                "Impact targets",
                "Source URI",
                "Created by",
            ]:
                await expect(page.get_by_text(label, exact=True).first).to_be_visible()
            await page.get_by_text("Decision options", exact=True).first.click()
            await page.get_by_label("Reason (optional)", exact=True).first.fill("Keep reviewed")
            await page.get_by_role("button", name="Keep", exact=True).first.click()
            await expect(page.get_by_role("button", name="Keep", exact=True)).to_have_count(1)
            await page.get_by_text("Decision options", exact=True).first.click()
            await page.get_by_label("Reason (optional)", exact=True).first.fill("Drop reviewed")
            await page.get_by_role("button", name="Drop", exact=True).first.click()
            await expect(page.get_by_role("button", name="Keep", exact=True)).to_have_count(0)
            await expect(page.get_by_label("Inbox total")).to_have_count(0)
            assert sum(request.endswith("/settle") for request in requests) == 2
            settlements = [
                await response.json()
                for response in responses
                if response.request.method == "POST"
                and urlsplit(response.url).path.endswith("/settle")
                and response.status == 200
            ]
            assert [result["item"]["settlement"]["reason"] for result in settlements] == [
                "Keep reviewed",
                "Drop reviewed",
            ]
            assert not any(response.status >= 400 for response in responses)

    asyncio.run(scenario())


@pytest.mark.parametrize("stale", [False, True])
def test_inbox_follows_empty_sealed_pages_and_never_shows_hidden_counts(stale):
    async def scenario():
        async with _home_browser(role="member", limited=True) as (page, base, _requests, responses):
            rejected = False
            list_reads = 0

            async def expire_cursor(route):
                nonlocal rejected, list_reads
                list_reads += 1
                if (
                    stale
                    and not rejected
                    and parse_qs(urlsplit(route.request.url).query).get("cursor")
                ):
                    rejected = True
                    await route.fulfill(status=422, json={"detail": "Cursor expired"})
                else:
                    await route.continue_()

            await page.route("**/api/inbox?*", expire_cursor)
            await page.goto(base + "/cayu/inbox?kind=approval", wait_until="networkidle")
            await expect(page.get_by_role("button", name="Approve", exact=True)).to_be_visible()
            await expect(
                page.get_by_text("Counts unavailable: your access is limited.", exact=True)
            ).to_be_visible()
            await expect(page.get_by_label("Inbox total")).to_have_count(0)
            await expect(page.locator("[data-inbox-item]")).to_have_count(1)
            await expect(
                page.locator("[data-inbox-item]").get_by_role("link", name="visible", exact=True)
            ).to_be_visible()
            assert list_reads <= 26, "Expired scoped pages get at most one bounded restart"
            pages = [
                await response.json()
                for response in responses
                if urlsplit(response.url).path == "/api/inbox" and response.status == 200
            ]
            assert rejected == stale
            assert pages[0]["items"] == [] and pages[0]["next_cursor"]
            assert any(
                item["session_id"] == "visible" for batch in pages for item in batch["items"]
            )
            assert not any("hidden-" in str(batch) for batch in pages)

    asyncio.run(scenario())


def test_home_start_uses_explicit_specialist_once_and_opens_session():
    async def scenario():
        async with _home_browser() as (page, base, _requests, _responses):
            submitted = []
            page.on(
                "request",
                lambda request: (
                    submitted.append(request.post_data_json)
                    if request.method == "POST"
                    and urlsplit(request.url).path == "/api/run"
                    and "last-event-id" not in request.headers
                    else None
                ),
            )
            await page.goto(base + "/cayu/?start=1", wait_until="networkidle")
            await expect(page.get_by_label("Specialist", exact=True)).to_have_value("")
            await page.get_by_label("Prompt", exact=True).fill("go")
            await expect(page.get_by_role("button", name="Start", exact=True)).to_be_disabled()
            await page.get_by_label("Specialist", exact=True).select_option("approver")
            await page.get_by_role("button", name="Start", exact=True).click()
            await expect(page).to_have_url(re.compile(r"/cayu/sessions/[^/?]+$"), timeout=20000)
            assert len(submitted) == 1
            assert submitted[0]["agent"] == "approver"
            assert submitted[0]["session_id"] in page.url

    asyncio.run(scenario())


def test_inbox_full_access_explicit_replacement_and_reason():
    async def scenario():
        async with _home_browser(role="developer", decisions=True) as (
            page,
            base,
            _requests,
            responses,
        ):
            bodies = []
            page.on(
                "request",
                lambda request: (
                    bodies.append(request.post_data_json)
                    if request.method == "POST" and urlsplit(request.url).path.endswith("/settle")
                    else None
                ),
            )
            await page.goto(base + "/cayu/inbox?kind=approval", wait_until="networkidle")
            await expect(
                page.get_by_label("Approve with changed arguments", exact=True)
            ).to_be_hidden()
            await page.get_by_text("Decision options", exact=True).click()
            await page.get_by_label("Approve with changed arguments", exact=True).check()
            await page.get_by_label("Changed arguments", exact=True).fill(
                '{"note":"reviewed replacement"}'
            )
            await page.get_by_label("Reason (optional)", exact=True).fill("Reviewed replacement")
            await page.get_by_text("Decision options", exact=True).click()
            await expect(page.get_by_label("Changed arguments", exact=True)).to_be_hidden()
            await page.get_by_text("Decision options", exact=True).click()
            await expect(page.get_by_label("Changed arguments", exact=True)).to_have_value(
                '{"note":"reviewed replacement"}'
            )
            await page.get_by_role("button", name="Approve", exact=True).click()
            await expect(page.get_by_role("button", name="Approve", exact=True)).to_have_count(0)
            assert bodies == [
                {
                    "decision": "approve",
                    "arguments": {"note": "reviewed replacement"},
                    "reason": "Reviewed replacement",
                }
            ]
            assert any(
                response.status == 200 and urlsplit(response.url).path.endswith("/settle")
                for response in responses
            )

    asyncio.run(scenario())


def test_inbox_mobile_keyboard_filters_and_theme():
    async def scenario():
        async with _home_browser(role="member", decisions=True, mobile=True) as (
            page,
            base,
            _requests,
            _responses,
        ):
            await page.goto(base + "/cayu/inbox", wait_until="networkidle")
            approvals = page.get_by_role("button", name=re.compile(r"^Approvals"))
            await approvals.focus()
            await page.keyboard.press("Enter")
            await expect(page).to_have_url(re.compile(r"kind=approval"))
            await expect(page.get_by_role("button", name="Approve", exact=True)).to_be_visible()
            assert await page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
            await page.get_by_role("button", name="Open sidebar", exact=True).click()
            theme = page.get_by_role("button", name="Theme: system", exact=True)
            await theme.focus()
            await page.keyboard.press("Enter")
            await expect(
                page.get_by_role("button", name="Theme: light", exact=True)
            ).to_be_visible()
            await page.keyboard.press("Escape")
            await expect(
                page.get_by_role("button", name="Open sidebar", exact=True)
            ).to_be_focused()

    asyncio.run(scenario())


def test_home_unknown_start_outcome_reobserves_without_second_submission():
    async def scenario():
        async with _home_browser() as (page, base, _requests, _responses):
            blocked = True
            submissions = []

            async def run_transport(route):
                if "last-event-id" not in route.request.headers:
                    submissions.append(route.request.post_data_json)
                    response = await route.fetch()
                    await route.fulfill(response=response, body=": accepted observer closed\n\n")
                elif blocked:
                    await route.fulfill(
                        status=503, json={"detail": "temporary observation failure"}
                    )
                else:
                    await route.continue_()

            async def reads(route):
                if blocked:
                    await route.fulfill(
                        status=503, json={"detail": "temporary observation failure"}
                    )
                else:
                    await route.continue_()

            await page.route("**/api/run", run_transport)
            await page.route("**/api/sessions/*/events?*", reads)
            await page.route("**/api/sessions/*/state", reads)
            await page.clock.install()
            await page.goto(base + "/cayu/?start=1", wait_until="networkidle")
            await page.get_by_label("Specialist", exact=True).select_option("approver")
            await page.get_by_label("Prompt", exact=True).fill("go")
            await page.get_by_role("button", name="Start", exact=True).click()
            for _ in range(90):
                if await page.locator('[data-mutation-transport-phase="transport_failed"]').count():
                    break
                await page.clock.run_for(1000)
                await asyncio.sleep(0.03)
            await expect(
                page.locator('[data-mutation-transport-phase="transport_failed"]')
            ).to_be_visible()
            await expect(page.get_by_role("button", name="Start", exact=True)).to_be_disabled()
            blocked = False
            await page.get_by_role("button", name="Retry observation", exact=True).click()
            await expect(page).to_have_url(re.compile(r"/cayu/sessions/[^/?]+$"), timeout=15000)
            assert len(submissions) == 1
            assert submissions[0]["session_id"] in page.url

    asyncio.run(scenario())


def test_home_unavailable_start_helper_does_not_mount_form(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples"))
    example = importlib.import_module("dashboard_behavior_live")

    async def scenario():
        async with _home_browser() as (page, base, requests, _responses):

            async def unavailable(route):
                response = await route.fetch()
                body = await response.json()
                body["capabilities"]["mutations"]["session_execution"] = {
                    "enabled": False,
                    "unavailable_reason": "unsupported",
                }
                await route.fulfill(response=response, json=body)

            await page.route("**/api/contract", unavailable)
            await example._assert_unavailable_run_page(page, base)
            assert "/api/run" not in requests

    asyncio.run(scenario())


def test_home_task_attention_is_independent_of_recent_task_sample():
    async def scenario():
        async with _home_browser(tasks=True) as (page, base, _requests, _responses):
            await page.goto(base + "/cayu/", wait_until="networkidle")
            section = page.get_by_role(
                "heading", name="Tasks needing attention", exact=True
            ).locator("../..")
            await expect(
                section.get_by_role("link", name=re.compile("Review older invoice"))
            ).to_be_visible()
            await expect(section.get_by_text("Needs operator review", exact=True)).to_be_visible()
            await expect(
                page.get_by_text("Showing the eight most recently updated tasks.", exact=False)
            ).to_be_visible()

    asyncio.run(scenario())


def test_home_idle_polling_helper_uses_bounded_server_reads(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples"))
    example = importlib.import_module("dashboard_behavior_live")

    async def scenario():
        async with _home_browser(decisions=True) as (page, base, _requests, _responses):

            async def context_factory(**_kwargs):
                return page.context

            observed = await example._exercise_idle_overview_polling(context_factory, base)
            assert "/api/sessions/overview/counts" in observed
            assert "/api/inbox/counts" in observed

    asyncio.run(scenario())


@pytest.mark.parametrize("agent", ["approver", "asker"])
@pytest.mark.parametrize("active_peer", [False, True])
def test_inbox_explicit_task_continuation_preserves_active_worker(agent, active_peer):
    async def scenario():
        async with _home_browser(role="member", worker_agent=agent, active_peer=active_peer) as (
            page,
            base,
            _requests,
            responses,
        ):
            await page.goto(base + "/cayu/inbox", wait_until="networkidle")
            await page.get_by_text("Decision options", exact=True).click()
            await page.get_by_label(
                "Continue an interrupted task with this decision.", exact=False
            ).check()
            await page.get_by_label("Reason (optional)", exact=True).fill(
                "Reviewed task continuation"
            )
            if agent == "asker":
                await page.get_by_label("Your answer", exact=True).fill("staging")
            async with page.expect_response(
                lambda response: (
                    response.request.method == "POST"
                    and urlsplit(response.url).path.endswith("/settle")
                )
            ) as response_info:
                await page.get_by_role(
                    "button", name="Approve" if agent == "approver" else "Send", exact=True
                ).click()
            response = await response_info.value
            assert response.request.post_data_json["task_continuation"] == "claim"
            assert "task_worker_id" not in response.request.post_data_json
            assert "task_handoff_id" not in response.request.post_data_json
            assert response.status == (409 if active_peer else 200), await response.text()
            if active_peer:
                await expect(page.get_by_role("alert").last).to_be_visible()
            else:
                assert (await response.json())["item"]["settlement"][
                    "reason"
                ] == "Reviewed task continuation"
                await expect(page.locator("[data-inbox-item]")).to_have_count(0)
            assert not any(response.status in (401, 403) for response in responses)

    asyncio.run(scenario())


def test_inbox_search_uses_server_membership_and_exact_filtered_counts():
    async def scenario():
        async with _home_browser(role="member", decisions=True) as (
            page,
            base,
            _requests,
            responses,
        ):
            await page.goto(base + "/cayu/inbox", wait_until="networkidle")
            await page.get_by_label("Search Inbox", exact=True).fill("correction")
            await page.get_by_role("button", name="Filter", exact=True).click()
            await expect(page).to_have_url(re.compile("q=correction"))
            await expect(page.locator("[data-inbox-item]")).to_have_count(1)
            await expect(
                page.get_by_role("button", name=re.compile("^Approvals 1$"))
            ).to_be_visible()
            await expect(page.get_by_role("button", name="Questions", exact=True)).to_be_visible()
            await page.reload(wait_until="networkidle")
            await expect(page.get_by_label("Search Inbox", exact=True)).to_have_value("correction")
            matching = [
                response
                for response in responses
                if parse_qs(urlsplit(response.url).query).get("q") == ["correction"]
            ]
            assert {urlsplit(response.url).path for response in matching} >= {
                "/api/inbox",
                "/api/inbox/counts",
            }

    asyncio.run(scenario())


@pytest.mark.parametrize("priced", [False, True])
def test_home_daily_usage_is_authoritative_and_excludes_eval_and_forks(priced):
    async def scenario():
        async with _home_browser(role="member", usage=True, priced=priced) as (
            page,
            base,
            _requests,
            responses,
        ):
            await page.goto(base + "/cayu/", wait_until="networkidle")
            summary = page.get_by_text("Last 24 hours:", exact=False)
            await expect(summary).to_contain_text("1 sessions · 0 failed")
            daily = [
                await response.json()
                for response in responses
                if urlsplit(response.url).path == "/api/sessions/overview/counts"
                and response.status == 200
            ]
            assert daily and daily[-1]["total_count"] == "1"
            counts = daily[-1]
            if priced:
                from decimal import Decimal

                assert Decimal(counts["cost"]["amount"]) > 0
                await expect(summary).to_contain_text(
                    f"{counts['cost']['amount']} {counts['cost']['currency']}"
                )
            else:
                tokens = int(counts["tokens"]["input"]) + int(counts["tokens"]["output"])
                assert tokens > 0 and counts["cost"] is None
                await expect(summary).to_contain_text(f"{tokens} tokens")
            await expect(page.locator('a[href$="/sessions/normal"]')).to_be_visible()
            await expect(
                page.locator('a[href$="/sessions/eval-trial"], a[href$="/sessions/forked"]')
            ).to_have_count(0)
            lists = [
                response
                for response in responses
                if urlsplit(response.url).path == "/api/sessions"
                and "ids" not in parse_qs(urlsplit(response.url).query)
            ]
            assert lists and all(
                parse_qs(urlsplit(response.url).query).get("include_eval") == ["false"]
                and parse_qs(urlsplit(response.url).query).get("include_forks") == ["false"]
                for response in lists
            )

    asyncio.run(scenario())


def test_inbox_stops_a_multi_page_cursor_cycle():
    async def scenario():
        async with _home_browser(role="member", limited=True) as (
            page,
            base,
            _requests,
            _responses,
        ):
            seen = []

            async def cycle(route):
                cursor = parse_qs(urlsplit(route.request.url).query).get("cursor", [None])[0]
                seen.append(cursor)
                await route.fulfill(
                    json={
                        "items": [],
                        "next_cursor": "B" if cursor == "A" else "A",
                        "has_more": True,
                        "issues": [],
                    }
                )

            await page.route("**/api/inbox?*", cycle)
            await page.goto(base + "/cayu/inbox", wait_until="domcontentloaded")
            await expect(
                page.get_by_text("The Inbox source did not advance. Refresh the list.", exact=True)
            ).to_be_visible()
            assert seen == [None, "A", "B"]

    asyncio.run(scenario())


def test_home_failure_link_opens_existing_filtered_event_log():
    async def scenario():
        async with _home_browser(failed=True) as (page, base, _requests, _responses):
            await page.goto(base + "/cayu/", wait_until="networkidle")
            link = page.locator('a[href*="/sessions/failed-session"]')
            await expect(link).to_have_count(1)
            assert parse_qs(urlsplit(await link.get_attribute("href")).query)["event_type"] == [
                "session.failed"
            ]
            await link.click()
            await expect(page).to_have_url(re.compile("event_type=session.failed"))
            await expect(page.get_by_label("Event type", exact=True)).to_have_value(
                "session.failed"
            )
            await expect(page.get_by_label("Event type", exact=True)).to_be_visible()
            await expect(
                page.get_by_role("button").filter(has_text="session.failed")
            ).to_be_visible()
            copied_url = page.url
            await page.goto(copied_url, wait_until="networkidle")
            await expect(page.get_by_label("Event type", exact=True)).to_be_visible()
            await expect(page.get_by_label("Event type", exact=True)).to_have_value(
                "session.failed"
            )

    asyncio.run(scenario())


def test_inbox_recovery_only_links_to_existing_session_evidence_flow():
    async def scenario():
        async with _home_browser(role="member", recovery=True) as (
            page,
            base,
            _requests,
            responses,
        ):
            await page.goto(base + "/cayu/inbox?kind=recovery", wait_until="networkidle")
            await expect(
                page.get_by_role("link", name="Recover in the session", exact=True)
            ).to_be_visible()
            await expect(page.locator("[data-inbox-item]").get_by_role("button")).to_have_count(0)
            await page.get_by_role("link", name="Recover in the session", exact=True).click()
            await expect(page).to_have_url(re.compile("/sessions/crashed$"))
            assert not any(
                response.request.method == "POST"
                and urlsplit(response.url).path.endswith("/settle")
                for response in responses
            )

    asyncio.run(scenario())


def test_inbox_long_empty_scan_can_continue_without_hidden_totals(monkeypatch):
    import cayu.server._inbox_access as inbox_access

    monkeypatch.setattr(inbox_access, "_SCAN_PAGES", 1)

    async def scenario():
        async with _home_browser(role="member", limited=True, hidden_count=30) as (
            page,
            base,
            _requests,
            _responses,
        ):
            await page.goto(base + "/cayu/inbox", wait_until="networkidle")
            await expect(
                page.get_by_role("button", name="Continue scanning", exact=True)
            ).to_be_visible()
            await expect(page.get_by_label("Inbox total")).to_have_count(0)
            await page.get_by_role("button", name="Continue scanning", exact=True).click()
            await expect(page.get_by_role("button", name="Approve", exact=True)).to_be_visible()
            await expect(page.locator("[data-inbox-item]")).to_have_count(1)
            await expect(page.get_by_label("Inbox total")).to_have_count(0)

    asyncio.run(scenario())


def test_inbox_filter_change_cannot_publish_an_old_inflight_page():
    async def scenario():
        async with _home_browser(role="member", decisions=True) as (
            page,
            base,
            _requests,
            _responses,
        ):
            started = asyncio.Event()
            release = asyncio.Event()

            async def delay_initial(route):
                if "q" in parse_qs(urlsplit(route.request.url).query):
                    await route.continue_()
                    return
                response = await route.fetch()
                started.set()
                await release.wait()
                # An obsolete read is expected to be aborted by query cancellation.
                with suppress(PlaywrightError):
                    await route.fulfill(response=response)

            await page.route("**/api/inbox?*", delay_initial)
            try:
                await page.goto(base + "/cayu/inbox", wait_until="domcontentloaded")
                await asyncio.wait_for(started.wait(), 10)
                await page.get_by_label("Search Inbox", exact=True).fill("correction")
                await page.get_by_role("button", name="Filter", exact=True).click()
                await expect(page.locator("[data-inbox-item]")).to_have_count(1)
                release.set()
                await page.wait_for_load_state("networkidle")
                await expect(page.get_by_role("button", name="Approve", exact=True)).to_be_visible()
                await expect(page.get_by_label("Your answer", exact=True)).to_have_count(0)
                await expect(page.locator("[data-inbox-item]")).to_have_count(1)
            finally:
                release.set()

    asyncio.run(scenario())


@pytest.mark.parametrize("boundary", ["delayed", "failed", "paused"])
def test_home_never_claims_attention_empty_without_complete_reads(boundary):
    async def scenario():
        async with _home_browser(decisions=True) as (page, base, _requests, _responses):
            started = asyncio.Event()
            release = asyncio.Event()
            scans = 0

            async def attention_read(route):
                nonlocal scans
                query = parse_qs(urlsplit(route.request.url).query)
                if "waiting_approval" not in query.get("overview_state", []):
                    await route.continue_()
                    return
                started.set()
                if boundary == "delayed":
                    await release.wait()
                    await route.continue_()
                elif boundary == "failed":
                    await route.fulfill(status=503, json={"detail": "Attention source unavailable"})
                elif scans < 25:
                    scans += 1
                    await route.fulfill(
                        json={
                            "sessions": [],
                            "overviews": [],
                            "total_count": None,
                            "next_cursor": f"scan-{scans}",
                            "has_more": True,
                        }
                    )
                else:
                    response = await page.request.get(
                        base
                        + "/api/sessions?include=overview&include_eval=false&include_forks=false&overview_state=waiting_approval"
                    )
                    await route.fulfill(response=response)

            await page.route("**/api/sessions?*", attention_read)
            try:
                await page.goto(base + "/cayu/", wait_until="domcontentloaded")
                await asyncio.wait_for(started.wait(), 10)
                if boundary == "delayed":
                    await expect(
                        page.get_by_text("Loading sessions needing attention…", exact=True)
                    ).to_be_visible()
                elif boundary == "failed":
                    await expect(
                        page.get_by_text("Attention could not be fully checked.", exact=True)
                    ).to_be_visible(timeout=15000)
                else:
                    await expect(
                        page.get_by_text(
                            "No matching attention sessions on the pages checked. Continue scanning.",
                            exact=True,
                        )
                    ).to_be_visible()
                await expect(
                    page.get_by_text("Nothing needs attention.", exact=True)
                ).to_have_count(0)
                await expect(
                    page.get_by_text("No sessions need attention.", exact=True)
                ).to_have_count(0)
                if boundary == "paused":
                    await page.get_by_role(
                        "button", name="Load more attention sessions", exact=True
                    ).click()
                    await expect(page.locator('a[href$="/sessions/approval"]')).to_be_visible()
            finally:
                release.set()

    asyncio.run(scenario())


@pytest.mark.parametrize("failed", [False, True])
def test_home_recent_filtered_empty_does_not_claim_agent_has_never_run(failed):
    async def scenario():
        async with _home_browser(decisions=not failed, failed=failed) as (
            page,
            base,
            _requests,
            _responses,
        ):
            await page.goto(base + "/cayu/", wait_until="networkidle")
            await expect(
                page.get_by_text("The loaded recent sessions need attention.", exact=True)
            ).to_be_visible()
            await expect(page.get_by_text("No sessions yet.", exact=False)).to_have_count(0)

    asyncio.run(scenario())


@pytest.mark.parametrize("failed", [False, True])
def test_home_recent_read_status_preserves_loaded_attention(failed):
    async def scenario():
        async with _home_browser(decisions=True) as (page, base, _requests, _responses):
            started = asyncio.Event()
            release = asyncio.Event()

            async def recent_read(route):
                query = parse_qs(urlsplit(route.request.url).query)
                if "overview_state" in query or "ids" in query:
                    await route.continue_()
                    return
                started.set()
                if failed:
                    await route.fulfill(status=503, json={"detail": "Recent source unavailable"})
                else:
                    await release.wait()
                    await route.continue_()

            await page.route("**/api/sessions?*", recent_read)
            try:
                await page.goto(base + "/cayu/", wait_until="domcontentloaded")
                await asyncio.wait_for(started.wait(), 10)
                await expect(
                    page.get_by_text(
                        "Recent sessions could not be fully checked."
                        if failed
                        else "Loading recent sessions…",
                        exact=True,
                    )
                ).to_be_visible(timeout=15000)
                await expect(page.locator('a[href$="/sessions/approval"]')).to_be_visible()
                await expect(page.get_by_text("No sessions yet.", exact=False)).to_have_count(0)
            finally:
                release.set()

    asyncio.run(scenario())


def test_inbox_explicit_refresh_recomputes_changed_phase_cursors():
    async def scenario():
        captured = {}
        async with _home_browser(
            decisions=True,
            knowledge=True,
            capture_runtime=lambda runtime: captured.update(runtime=runtime),
        ) as (page, base, _requests, _responses):
            await page.goto(base + "/cayu/inbox", wait_until="networkidle")
            await page.get_by_text("Advanced filters", exact=True).click()
            await page.get_by_label("Inbox page size", exact=True).select_option("25")
            await page.get_by_role("button", name="Load more", exact=True).click()
            await expect(page.locator("[data-inbox-item]")).to_have_count(2)
            release = asyncio.Event()

            async def hold_refresh(route):
                await release.wait()
                await route.continue_()

            await page.route("**/api/inbox?*", hold_refresh)
            for index in range(26):
                async for _event in captured["runtime"].run(
                    RunRequest(
                        agent_name="approver",
                        session_id=f"arrived-{index}",
                        messages=[Message.text("user", "go")],
                    )
                ):
                    pass
            release.set()
            await page.wait_for_timeout(5500)
            await expect(page.locator("[data-inbox-item]")).to_have_count(2)
            await page.get_by_role("button", name="Refresh actions", exact=True).click()
            await expect(
                page.get_by_text("The Inbox source did not advance. Refresh the list.", exact=True)
            ).to_have_count(0)
            await expect(page.locator("[data-inbox-item]")).to_have_count(25, timeout=10000)
            await page.get_by_role("button", name="Load more", exact=True).click()
            await expect(page.locator("[data-inbox-item]")).to_have_count(3)
            await page.get_by_role("button", name="Load more", exact=True).click()
            await expect(page.locator("[data-inbox-item]")).to_have_count(2)
            await page.wait_for_timeout(5500)
            await expect(page.locator("[data-inbox-item]")).to_have_count(2)

    asyncio.run(scenario())


def test_inbox_knowledge_full_proposal_precedes_keep_decision():
    async def scenario():
        async with _home_browser(role="member", knowledge=True, long_knowledge=True) as (
            page,
            base,
            _requests,
            responses,
        ):
            await page.goto(base + "/cayu/inbox?kind=knowledge", wait_until="networkidle")
            await expect(page.get_by_text("FINAL REVIEW CONDITION", exact=False)).to_be_visible()
            card = page.locator("[data-inbox-item]").filter(has_text="FINAL REVIEW CONDITION")
            await card.get_by_role("button", name="Keep", exact=True).click()
            await expect(card).to_have_count(0)
            assert any(
                response.status == 200 and urlsplit(response.url).path.endswith("/settle")
                for response in responses
            )

    asyncio.run(scenario())


def test_home_daily_summary_survives_window_rollover():
    async def scenario():
        async with _home_browser() as (page, base, _requests, _responses):
            await page.clock.install()
            release = asyncio.Event()
            started = asyncio.Event()

            async def hold_new_window(route):
                started.set()
                await release.wait()
                await route.continue_()

            await page.goto(base + "/cayu/", wait_until="networkidle")
            await expect(page.get_by_text("Last 24 hours:", exact=False)).to_be_visible()
            await page.route("**/api/sessions/overview/counts?*", hold_new_window)
            try:
                await page.clock.run_for(61000)
                await asyncio.wait_for(started.wait(), 10)
                await expect(page.get_by_text("Last 24 hours:", exact=False)).to_be_visible()
            finally:
                release.set()

    asyncio.run(scenario())


def test_dashboard_shell_navigation_stays_named_when_inbox_count_exists(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples"))
    example = importlib.import_module("dashboard_behavior_live")

    async def scenario():
        async with _home_browser(decisions=True) as (page, base, _requests, _responses):
            await example._exercise_dashboard_shell(page, base)

    asyncio.run(scenario())


def test_inbox_child_actions_refresh_while_queue_is_idle():
    async def scenario():
        async with _home_browser() as (page, base, requests, _responses):
            await page.clock.install()
            await page.goto(base + "/cayu/inbox", wait_until="networkidle")
            before = requests.count("/api/pending-actions")
            assert before > 0
            await page.clock.run_for(5500)
            await page.wait_for_load_state("networkidle")
            assert requests.count("/api/pending-actions") > before

    asyncio.run(scenario())


@pytest.mark.parametrize("boundary", ["unavailable", "changed"])
def test_inbox_knowledge_requires_full_exact_revision_before_deciding(boundary):
    async def scenario():
        async with _home_browser(role="member", knowledge=True) as (
            page,
            base,
            _requests,
            _responses,
        ):

            async def source(route):
                if boundary == "unavailable":
                    await route.fulfill(status=503, json={"detail": "Proposal source unavailable"})
                else:
                    response = await route.fetch()
                    body = await response.json()
                    body["revision"] += 1
                    await route.fulfill(response=response, json=body)

            await page.route("**/api/knowledge/pending/*", source)
            await page.goto(base + "/cayu/inbox?kind=knowledge", wait_until="networkidle")
            await expect(page.get_by_role("button", name="Keep", exact=True).first).to_be_disabled()
            await expect(page.get_by_role("button", name="Drop", exact=True).first).to_be_disabled()

    asyncio.run(scenario())


def test_workflow_request_contract_includes_shared_inbox_badge(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples"))
    example = importlib.import_module("dashboard_behavior_live")

    async def scenario():
        runtime, _provider, sessions, tasks, directory = await example._seed_app()
        try:
            async with _home_browser(
                seeded_runtime=runtime,
                dashboard_config={"priceBook": example._dashboard_price_book()},
            ) as (page, base, _requests, _responses):
                aborted = []
                page.on(
                    "requestfailed",
                    lambda request: (
                        aborted.append(request.url)
                        if request.method == "POST"
                        and urlsplit(request.url).path.endswith("/topology")
                        and request.failure == "net::ERR_ABORTED"
                        else None
                    ),
                )
                await example._exercise_workflow(page, base, sessions, tasks, aborted)
        finally:
            directory.cleanup()

    asyncio.run(scenario())


@pytest.mark.parametrize("panel", ["inbox", "children"])
def test_inbox_source_issues_keep_distinct_session_inspection_links(panel):
    async def scenario():
        async with _home_browser(decisions=True) as (page, base, _requests, _responses):

            async def issues(route):
                response = await route.fetch()
                body = await response.json()
                body["issues"] = [
                    {
                        "code": "pending_action_source_incomplete",
                        "detail": "Open the session directly to inspect or resolve it.",
                        "session_id": session,
                        "agent_name": "approver",
                    }
                    for session in ["approval", "question"]
                ]
                await route.fulfill(response=response, json=body)

            path = "inbox" if panel == "inbox" else "pending-actions"
            await page.route(f"**/api/{path}?*", issues)
            await page.goto(base + "/cayu/inbox", wait_until="networkidle")
            for session in ["approval", "question"]:
                await expect(
                    page.get_by_role("link", name=f"Inspect session {session}", exact=True)
                ).to_be_visible()
            await page.get_by_role("link", name="Inspect session approval", exact=True).click()
            await expect(page).to_have_url(re.compile(r"/cayu/sessions/approval$"))

    asyncio.run(scenario())


@pytest.mark.parametrize("role", ["admin", "member"])
@pytest.mark.parametrize("width", [390, 1440])
def test_run_execution_is_independent_of_overview_read_support(role, width):
    from cayu.sessions.store import SessionStore

    def unsupported(runtime):
        class NoOverviewStore(type(runtime.session_store)):
            invocation_lifecycle_command_version = 1
            terminal_interaction_publication_version = 1
            read_usage_accounting = SessionStore.read_usage_accounting

        runtime.session_store.__class__ = NoOverviewStore

    async def scenario():
        async with _home_browser(role=role, capture_runtime=unsupported) as (
            page,
            base,
            requests,
            _responses,
        ):
            await page.set_viewport_size({"width": width, "height": 900})
            await page.goto(base + "/cayu/run", wait_until="networkidle")
            await expect(page.get_by_test_id("dashboard-capability-unavailable")).to_be_visible()
            await expect(page.locator("main header")).to_have_count(1)
            await expect(page.locator("main div.overflow-y-auto")).to_have_count(1)
            if width == 390:
                opener = page.get_by_role("button", name="Open sidebar", exact=True)
                await expect(opener).to_have_count(1)
                await opener.click()
                await page.keyboard.press("Escape")
                await expect(opener).to_be_focused()
            if role == "admin":
                await page.get_by_label("Specialist", exact=True).select_option("approver")
                await page.get_by_label("Prompt", exact=True).fill("go")
                await page.get_by_role("button", name="Start", exact=True).click()
                await expect(page).to_have_url(re.compile(r"/cayu/sessions/[^/?]+$"))
            else:
                await expect(page.get_by_label("Prompt", exact=True)).to_have_count(0)
                assert "/api/run" not in requests
            assert "/api/sessions/overview/counts" not in requests

    asyncio.run(scenario())


def test_shared_shell_preserves_terminal_waiting_and_home_polling_budgets(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples"))
    example = importlib.import_module("dashboard_behavior_live")
    asyncio.run(example._run_polling_budget_contract())


def test_eval_release_acceptance_starts_through_current_shared_form(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples"))
    behavior = importlib.import_module("dashboard_behavior_live")
    example = importlib.import_module("evals_release_acceptance_live")
    monkeypatch.setattr(example, "AGENT_NAME", behavior.POLLING_AGENT_NAME)

    async def scenario():
        runtime = await behavior._seed_polling_budget_app()
        async with _home_browser(seeded_runtime=runtime) as (page, base, requests, _responses):
            session_id = await example._start_source_session(page, base)
            assert session_id
            assert requests.count("/api/run") == 1
            await expect(page).to_have_url(re.compile(r"/cayu/sessions/[^/?]+$"))

    asyncio.run(scenario())


def test_eval_release_acceptance_rejects_failed_source_session(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples"))
    example = importlib.import_module("evals_release_acceptance_live")
    monkeypatch.setattr(example, "AGENT_NAME", "failing")

    async def scenario():
        async with _home_browser(failed=True) as (page, base, _requests, _responses):
            with pytest.raises(RuntimeError, match="source session must complete"):
                await example._start_source_session(page, base)

    asyncio.run(scenario())


@pytest.mark.parametrize("surface", ["inbox?kind=approval", "sessions/approval"])
def test_approval_role_refresh_exposes_retained_override_and_allows_original(surface):
    async def scenario():
        async with _home_browser(role="developer", decisions=True) as (page, base, _, _responses):
            mode = "developer"
            bodies = []

            async def identity(route):
                if mode == "temporary":
                    await route.fulfill(status=503, json={"detail": "temporary"})
                    return
                response = await route.fetch()
                body = await response.json()
                body.update(role=mode, full_access=mode == "developer")
                await route.fulfill(response=response, json=body)

            await page.route("**/api/app", identity)
            page.on(
                "request",
                lambda r: (
                    bodies.append(r.post_data_json)
                    if r.method == "POST" and urlsplit(r.url).path.endswith("/settle")
                    else None
                ),
            )
            await page.goto(base + "/cayu/" + surface, wait_until="networkidle")
            await page.get_by_text("Decision options", exact=True).click()
            await page.get_by_label("Approve with changed arguments", exact=True).check()
            await page.get_by_label("Changed arguments", exact=True).fill('{"note":"kept draft"}')
            await page.get_by_label("Reason (optional)", exact=True).fill("Reviewed original")
            mode = "temporary"
            await page.evaluate("window.dispatchEvent(new Event('visibilitychange'))")
            await expect(page.get_by_test_id("dashboard-identity-refresh-warning")).to_be_visible()
            await expect(page.get_by_label("Changed arguments", exact=True)).to_have_value(
                '{"note":"kept draft"}'
            )
            mode = "member"
            await page.evaluate("window.dispatchEvent(new Event('visibilitychange'))")
            await expect(
                page.get_by_label("Approve with changed arguments", exact=True)
            ).to_be_hidden()
            original = page.get_by_role("button", name="Use original arguments", exact=True)
            await expect(original).to_be_visible()
            await expect(page.get_by_role("button", name="Approve", exact=True)).to_be_disabled()
            assert bodies == []
            mode = "developer"
            await page.evaluate("window.dispatchEvent(new Event('visibilitychange'))")
            await expect(page.get_by_label("Changed arguments", exact=True)).to_have_value(
                '{"note":"kept draft"}'
            )
            mode = "member"
            await page.evaluate("window.dispatchEvent(new Event('visibilitychange'))")
            await original.click()
            await page.get_by_role("button", name="Approve", exact=True).click()
            await expect(page.get_by_role("button", name="Approve", exact=True)).to_have_count(0)
            assert bodies == [{"decision": "approve", "reason": "Reviewed original"}]

    asyncio.run(scenario())


@pytest.mark.parametrize("surface", ["inbox?kind=approval", "sessions/approval"])
def test_pending_approval_keeps_dispatched_arguments_during_role_refresh(surface):
    async def scenario():
        async with _home_browser(role="developer", decisions=True) as (page, base, _, _responses):
            member = False
            dispatched = asyncio.Event()
            release = asyncio.Event()
            bodies = []

            async def identity(route):
                response = await route.fetch()
                body = await response.json()
                if member:
                    body.update(role="member", full_access=False)
                await route.fulfill(response=response, json=body)

            async def hold_decision(route):
                bodies.append(route.request.post_data_json)
                dispatched.set()
                await release.wait()
                await route.continue_()

            await page.route("**/api/app", identity)
            await page.route("**/api/inbox/*/settle", hold_decision)
            await page.goto(base + "/cayu/" + surface, wait_until="networkidle")
            await page.get_by_text("Decision options", exact=True).click()
            await page.get_by_label("Approve with changed arguments", exact=True).check()
            await page.get_by_label("Changed arguments", exact=True).fill('{"note":"dispatched"}')
            await page.get_by_role("button", name="Approve", exact=True).click()
            await asyncio.wait_for(dispatched.wait(), 5)
            try:
                member = True
                await page.evaluate("window.dispatchEvent(new Event('visibilitychange'))")
                await expect(
                    page.get_by_role("button", name="Use original arguments", exact=True)
                ).to_be_disabled()
                await expect(
                    page.get_by_role("button", name="Approve", exact=True)
                ).to_be_disabled()
                assert bodies == [
                    {"decision": "approve", "arguments": {"note": "dispatched"}, "reason": None}
                ]
            finally:
                release.set()
            await expect(page.get_by_role("button", name="Approve", exact=True)).to_have_count(0)
            assert len(bodies) == 1

    asyncio.run(scenario())
