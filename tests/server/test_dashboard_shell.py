"""Focused shell acceptance through the packaged dashboard and signed proxy auth."""

from __future__ import annotations

import asyncio
import importlib
import os
import re
import socket
from contextlib import asynccontextmanager
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlsplit

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("playwright.async_api")

import uvicorn
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import FastAPI
from playwright.async_api import async_playwright, expect
from tests.server.test_proxy_identity import FakeJwks, _as, _auth, _token

from cayu.applications import CayuApp
from cayu.events import Event, EventType
from cayu.server import ServerConfig, create_server
from cayu.server.config import DashboardConfig
from cayu.sessions.records import SessionIdentity, SessionStatus
from cayu.sessions.requests import RunRequest
from cayu.tasks import InMemoryTaskStore

pytestmark = pytest.mark.skipif(
    os.environ.get("CAYU_DASHBOARD_SHELL_LIVE") != "1",
    reason="Opt-in packaged dashboard Chromium acceptance.",
)


@asynccontextmanager
async def _shell_browser(*, role="admin", nested=False, mobile=False, decisions=False):
    key = ec.generate_private_key(ec.SECP256R1())
    auth = _auth(FakeJwks(key))
    runtime = CayuApp(task_store=InMemoryTaskStore())
    if decisions:
        await _seed_decisions(runtime)
    app = create_server(
        runtime,
        config=ServerConfig.protected(
            auth,
            deployment_name="Shell acceptance",
            dashboard=DashboardConfig(directory=os.environ.get("CAYU_DASHBOARD_SHELL_DIRECTORY")),
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


def test_dashboard_shell_admin_nested_navigation_and_search(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples"))
    example = importlib.import_module("dashboard_behavior_live")

    async def scenario():
        async with _shell_browser(nested=True) as (page, base, requests, responses):
            await example._exercise_dashboard_shell(page, base)
            sidebar = page.get_by_role("complementary", name="Sidebar")
            await expect(sidebar).to_be_visible()
            await expect(sidebar.get_by_text("Shell acceptance", exact=True)).to_be_visible()
            await expect(
                page.locator("main header").get_by_text("Shell acceptance", exact=True)
            ).to_be_visible()
            await expect(sidebar.get_by_text("v" + version("cayu"), exact=True)).to_be_visible()
            await page.get_by_role("button", name="Hide sidebar", exact=True).click()
            await expect(sidebar).not_to_be_visible()
            await page.keyboard.press("ControlOrMeta+k")
            await expect(sidebar).to_be_visible()
            navigation = sidebar.get_by_role("navigation", name="Main")
            for label in ("Home", "Inbox", "Sessions", "Tasks", "Usage", "Specialists"):
                await expect(navigation.get_by_role("link", name=label, exact=True)).to_be_visible()
            await expect(navigation.get_by_role("link", name="Tools", exact=True)).to_have_count(0)
            search = sidebar.get_by_label("Search sessions", exact=True)
            await page.keyboard.press("ControlOrMeta+k")
            await expect(search).to_be_focused()
            await search.fill("shell search")
            await search.press("Enter")
            await expect(page).to_have_url(re.compile(r"/cayu/sessions\?.*"))
            assert "shell" in page.url
            for path in ("/pending-actions", "/agents", "/sessions?q=preserved"):
                await page.goto(base + "/cayu" + path, wait_until="networkidle")
                await expect(sidebar).to_be_visible()
                assert path in page.url
            assert "/product/api/app" in requests
            assert not any(response.status in (401, 403) for response in responses)

    asyncio.run(scenario())


def test_dashboard_shell_member_hides_full_access_routes_before_page_reads():
    async def scenario():
        async with _shell_browser(role="member") as (page, base, requests, responses):
            await page.goto(base + "/cayu/pending-actions", wait_until="networkidle")
            navigation = page.get_by_role("navigation", name="Main")
            await expect(navigation).to_be_visible()
            for label in ("Home", "Inbox", "Sessions", "Tasks", "Usage"):
                await expect(navigation.get_by_role("link", name=label, exact=True)).to_be_visible()
            for label in ("Evals", "System", "Run", "Start a session"):
                await expect(navigation.get_by_role("link", name=label, exact=True)).to_have_count(
                    0
                )
            for path in ("/evals", "/system", "/run"):
                requests.clear()
                await page.goto(base + "/cayu" + path, wait_until="networkidle")
                await expect(
                    page.get_by_role("status").filter(
                        has_text="Your role can read and decide, but this action requires full access."
                    )
                ).to_be_visible()
                assert not any(
                    request.startswith(("/api/evals", "/api/system", "/api/run"))
                    for request in requests
                ), requests
            assert not any(response.status in (401, 403) for response in responses)

    asyncio.run(scenario())


def test_dashboard_shell_theme_mobile_keyboard_and_csp():
    async def scenario():
        async with _shell_browser(mobile=True) as (page, base, _requests, responses):
            await page.add_init_script(
                "window.shellCspViolations=[]; document.addEventListener('securitypolicyviolation', "
                "event=>window.shellCspViolations.push(event.violatedDirective));"
            )
            await page.goto(base + "/cayu/sessions", wait_until="networkidle")
            await expect(page.locator("html")).to_have_class(re.compile(r"\bdark\b"))
            await _assert_shell_contrast(page)
            opener = page.get_by_role("button", name="Open sidebar", exact=True)
            await expect(opener).to_be_visible()
            await opener.focus()
            await page.keyboard.press("Enter")
            sidebar = page.get_by_role("dialog", name="Sidebar")
            await expect(sidebar).to_be_visible()
            theme = page.get_by_role("button", name="Theme: system", exact=True)
            await theme.click()
            await expect(
                page.get_by_role("button", name="Theme: light", exact=True)
            ).to_be_visible()
            await expect(page.locator("html")).not_to_have_class(re.compile(r"\bdark\b"))
            await _assert_shell_contrast(page)
            await page.get_by_role("button", name="Theme: light", exact=True).click()
            await expect(page.locator("html")).to_have_class(re.compile(r"\bdark\b"))
            await page.reload(wait_until="networkidle")
            await expect(page.locator("html")).to_have_class(re.compile(r"\bdark\b"))
            await page.get_by_role("button", name="Open sidebar", exact=True).click()
            await expect(page.get_by_role("button", name="Theme: dark", exact=True)).to_be_visible()
            await page.get_by_role("button", name="Theme: dark", exact=True).click()
            await page.emulate_media(color_scheme="light")
            await expect(page.locator("html")).not_to_have_class(re.compile(r"\bdark\b"))
            await page.keyboard.press("Escape")
            await expect(opener).to_be_focused()
            assert await page.evaluate("window.shellCspViolations") == []
            document = next(
                response for response in responses if response.request.is_navigation_request()
            )
            policy = (await document.all_headers())["content-security-policy"]
            assert "'unsafe-inline'" not in policy.split("script-src ", 1)[1].split(";", 1)[0]

    asyncio.run(scenario())


def test_dashboard_shell_contract_mismatch_never_mounts_page_reads():
    async def scenario():
        async with _shell_browser() as (page, base, requests, _responses):

            async def mismatch(route):
                response = await route.fetch()
                body = await response.json()
                body["contract_version"] = -1
                await route.fulfill(response=response, json=body)

            await page.route("**/api/contract", mismatch)
            await page.goto(base + "/cayu/sessions", wait_until="networkidle")
            await expect(page.get_by_test_id("dashboard-contract-gate")).to_be_visible()
            assert not any(request.startswith("/api/sessions") for request in requests), requests

    asyncio.run(scenario())


async def _seed_decisions(app):
    """Persist ordinary paused records; browser requests traverse all server projections."""
    identity = {
        "model_step_id": "mstep_" + "1" * 32,
        "model_attempt_id": "matt_" + "2" * 32,
        "tool_round_id": "tround_" + "3" * 32,
    }
    for kind in ("approval", "answer"):
        session_id = "shell-" + kind
        session = await app.session_store.create(
            RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        call = {
            "tool_call_id": "shell-call",
            "tool_name": "shell-tool",
            "arguments": {},
            "policy_decision": "require_approval" if kind == "approval" else None,
            "reason": None,
            "metadata": {},
            "active_taint_labels": [],
        }
        pending = {
            **identity,
            "tool_call_id": "shell-call",
            "tool_name": "shell-tool",
            "arguments": {},
            "agent_name": "assistant",
            "tool_calls": [call],
        }
        if kind == "approval":
            pending.update(approval_id="shell-approval", publish_arguments=True, reason=None)
            checkpoint = {
                "pending_tool_approval": pending,
                "pending_tool_round": {
                    **identity,
                    "agent_name": "assistant",
                    "tool_calls": [call],
                    "policy_state": "planned",
                    "policy_context_version": 1,
                },
            }
            event_type = EventType.TOOL_CALL_APPROVAL_REQUESTED
            payload = {
                **identity,
                "approval_id": "shell-approval",
                "tool_call_id": "shell-call",
                "approval": pending,
            }
        else:
            pending.update(
                input_id="shell-input",
                question="Choose the next step",
                options=["Continue", "Wait"],
                session_id=session_id,
                session_instance_id=session.instance_id,
                source_interaction_id="interaction:shell-answer",
                source_run_epoch=session.run_epoch,
                execution_profile_fingerprint="e" * 64,
            )
            checkpoint = {"pending_user_input": pending}
            event_type = EventType.SESSION_AWAITING_USER_INPUT
            payload = {
                **identity,
                "input_id": "shell-input",
                "tool_call_id": "shell-call",
                "question": "Choose the next step",
                "options": ["Continue", "Wait"],
            }
        await app.session_store.update_status(session_id, SessionStatus.INTERRUPTED)
        await app.session_store.append_events(
            session_id,
            [
                Event(
                    id="shell-event-" + kind,
                    type=event_type,
                    session_id=session_id,
                    agent_name="assistant",
                    tool_name="shell-tool",
                    payload=payload,
                )
            ],
        )
        await app.session_store.checkpoint(session_id, checkpoint)


def test_dashboard_shell_member_keeps_decisions_without_full_access_widgets():
    async def scenario():
        async with _shell_browser(role="member", decisions=True) as (
            page,
            base,
            requests,
            responses,
        ):
            await page.goto(base + "/cayu/sessions/shell-approval", wait_until="networkidle")
            await expect(page.get_by_role("button", name="Approve", exact=True)).to_be_enabled()
            await expect(page.get_by_role("button", name="Deny", exact=True)).to_be_enabled()
            await page.goto(base + "/cayu/sessions/shell-answer", wait_until="networkidle")
            await page.get_by_placeholder("Answer the paused session...").fill("Continue")
            await expect(
                page.get_by_role("button", name="Submit Answer", exact=True)
            ).to_be_enabled()
            assert not any("/browser-control" in path or "/evals" in path for path in requests), (
                requests
            )
            assert not any(response.status in (401, 403) for response in responses)

    asyncio.run(scenario())


def test_dashboard_shell_identity_failure_never_grants_page_access():
    async def scenario():
        async with _shell_browser() as (page, base, requests, _responses):
            await page.route(
                "**/api/app",
                lambda route: route.fulfill(status=503, json={"detail": "identity unavailable"}),
            )
            await page.goto(base + "/cayu/sessions", wait_until="networkidle")
            await expect(page.get_by_test_id("dashboard-identity-gate")).to_be_visible()
            assert not any(request.startswith("/api/sessions") for request in requests), requests
            await expect(page.get_by_role("link", name="Run", exact=True)).to_have_count(0)

    asyncio.run(scenario())


async def _assert_shell_contrast(page):
    ratios = await page.evaluate("""() => {
      const canvas = document.createElement('canvas'); canvas.width = canvas.height = 1;
      const ctx = canvas.getContext('2d', {willReadFrequently: true});
      const sample = document.createElement('span'); document.body.append(sample);
      function luminance(variable) {
        sample.style.color = `var(--${variable})`;
        ctx.clearRect(0, 0, 1, 1); ctx.fillStyle = getComputedStyle(sample).color;
        ctx.fillRect(0, 0, 1, 1);
        const rgb = [...ctx.getImageData(0, 0, 1, 1).data].slice(0, 3).map(v => {
          v /= 255; return v <= .04045 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4;
        });
        return rgb[0] * .2126 + rgb[1] * .7152 + rgb[2] * .0722;
      }
      const pairs = [['foreground', 'background'], ['muted-foreground', 'background'],
        ['sidebar-foreground', 'sidebar-background'], ['muted-foreground', 'sidebar-background']];
      const ratios = Object.fromEntries(pairs.map(([fg,bg]) => {
        const a=luminance(fg), b=luminance(bg);
        return [`${fg}/${bg}`, (Math.max(a,b)+.05)/(Math.min(a,b)+.05)];
      })); sample.remove(); return ratios;
    }""")
    assert all(ratio >= 4.5 for ratio in ratios.values()), ratios


def test_dashboard_shell_unconfigured_evals_matches_capability_contract(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples"))
    example = importlib.import_module("dashboard_behavior_live")

    async def scenario():
        async with _shell_browser() as (page, base, _requests, _responses):
            await example._exercise_unconfigured_evals_shell(page, base)

    asyncio.run(scenario())


def test_identity_refresh_preserves_drafts_but_revokes_stale_authority():
    async def scenario():
        async with _shell_browser() as (page, base, _requests, _responses):
            mode = "valid"

            async def identity(route):
                if mode == "network":
                    await route.abort("failed")
                elif mode in {"408", "429", "503", "403"}:
                    await route.fulfill(
                        status=int(mode),
                        json={
                            "detail": {"code": "role_forbidden"} if mode == "403" else "unavailable"
                        },
                    )
                else:
                    response = await route.fetch()
                    body = await response.json()
                    if mode == "malformed":
                        body["full_access"] = "true"
                    elif mode == "member":
                        body.update(role="member", full_access=False)
                    await route.fulfill(response=response, json=body)

            await page.route("**/api/app", identity)
            prompt = page.locator("#run-prompt")
            for failure in ("408", "429", "503", "network"):
                mode = "valid"
                await page.goto(base + "/cayu/run", wait_until="networkidle")
                await prompt.fill("Unsaved approval investigation")
                mode = failure
                await page.evaluate("window.dispatchEvent(new Event('visibilitychange'))")
                await expect(
                    page.get_by_test_id("dashboard-identity-refresh-warning")
                ).to_be_visible()
                await expect(prompt).to_have_value("Unsaved approval investigation")
            mode = "403"
            await page.evaluate("window.dispatchEvent(new Event('visibilitychange'))")
            await expect(page.get_by_test_id("dashboard-identity-gate")).to_be_visible()
            await expect(prompt).to_have_count(0)
            mode = "503"
            await page.get_by_role("button", name="Retry identity check").click()
            await expect(page.get_by_test_id("dashboard-identity-gate")).to_be_visible()
            await expect(prompt).to_have_count(0)
            mode = "valid"
            await page.get_by_role("button", name="Retry identity check").click()
            await expect(prompt).to_be_visible()
            mode = "malformed"
            await page.evaluate("window.dispatchEvent(new Event('visibilitychange'))")
            await expect(page.get_by_test_id("dashboard-identity-gate")).to_be_visible()
            mode = "valid"
            await page.get_by_role("button", name="Retry identity check").click()
            await expect(prompt).to_be_visible()
            mode = "member"
            await page.evaluate("window.dispatchEvent(new Event('visibilitychange'))")
            await expect(prompt).to_have_count(0)
            await expect(
                page.get_by_role("status").filter(has_text="Your role can read and decide")
            ).to_be_visible()

    asyncio.run(scenario())


def test_unavailable_run_page_preserves_shell_search(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples"))
    example = importlib.import_module("dashboard_behavior_live")

    async def scenario():
        async with _shell_browser() as (page, base, _requests, _responses):

            async def disable_execution(route):
                response = await route.fetch()
                body = await response.json()
                body["capabilities"]["mutations"]["session_execution"] = {
                    "enabled": False,
                    "unavailable_reason": "unsupported",
                }
                await route.fulfill(response=response, json=body)

            await page.route("**/api/contract", disable_execution)
            await example._assert_unavailable_run_page(page, base)
            await expect(page.get_by_placeholder("Search sessions")).to_be_visible()
            await expect(page.get_by_role("main").get_by_role("textbox")).to_have_count(0)

    asyncio.run(scenario())


def test_session_discovery_search_is_independent_of_shell_search(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples"))
    example = importlib.import_module("dashboard_behavior_live")

    async def scenario():
        async with _shell_browser() as (page, base, _requests, _responses):
            await page.goto(base + "/cayu/sessions", wait_until="networkidle")
            sidebar = page.get_by_role("complementary", name="Sidebar")
            global_search = sidebar.get_by_label("Search sessions", exact=True)
            await global_search.fill("Keep my global search")
            search = example._session_discovery_search(page)
            async with page.expect_response(
                lambda response: (
                    "/api/sessions/summary?" in response.url and "q=filter-marker" in response.url
                )
            ) as response_info:
                await search.fill("filter-marker")
            assert (await response_info.value).status == 200
            await expect(page).to_have_url(re.compile(r"[?&]q=filter-marker(?:&|$)"))
            await expect(search).to_have_value("filter-marker")
            await expect(global_search).to_have_value("Keep my global search")

    asyncio.run(scenario())
