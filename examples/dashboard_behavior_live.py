"""Verified browser contract for the packaged Cayu dashboard."""

from __future__ import annotations

import asyncio
import json
import os
import re
import socket
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlsplit

import uvicorn
from playwright.async_api import (
    BrowserContext,
    Page,
    Request,
    async_playwright,
    expect,
)

from _live_checks import require, require_equal
from cayu import (
    AgentSpec,
    CayuApp,
    Event,
    EventType,
    Message,
    MessageRole,
    RunRequest,
    TextPart,
    ThinkingPart,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
)
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import EventQuery, InMemorySessionStore, SessionIdentity, SessionStatus
from cayu.server import create_server

if TYPE_CHECKING:
    from starlette.types import ASGIApp, Receive, Scope, Send
    from starlette.types import Message as AsgiMessage

SESSION_ID = "dashboard-contract-session"
APPROVAL_SESSION_ID = "dashboard-contract-approval"
INTERRUPT_SESSION_ID = "dashboard-contract-interrupt"
INTERRUPT_FAILURE_SESSION_ID = "dashboard-contract-interrupt-failure"
RESUME_INTERRUPT_SESSION_ID = "dashboard-contract-resume-interrupt"
REOBSERVE_SESSION_ID = "dashboard-contract-reobserve"
AGENT_NAME = "dashboard-contract-agent"
PROVIDER_NAME = "contract-provider"
MODEL_NAME = "contract-model"
PAYLOAD_MARKER = "dashboard-contract-usage"
EVIDENCE_PREFIX = "CAYU_NIGHTLY_EVIDENCE="


class DashboardContractProvider(ModelProvider):
    name = "contract-provider"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self.direct_requests: list[ModelRequest] = []
        self.direct_completions = 0
        self.recovery_requests: list[ModelRequest] = []
        self.replay_markers: list[str] = []
        self._direct_started = asyncio.Condition()
        self._direct_releases: asyncio.Queue[None] = asyncio.Queue()
        self._replay_releases: asyncio.Queue[str] = asyncio.Queue()

    async def wait_for_direct_requests(self, count: int) -> None:
        async with self._direct_started:
            await asyncio.wait_for(
                self._direct_started.wait_for(lambda: len(self.direct_requests) >= count),
                timeout=10,
            )

    def release_direct(self) -> None:
        self._direct_releases.put_nowait(None)

    def release_after_replay(self, marker: str) -> None:
        self._replay_releases.put_nowait(marker)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        request_text = "\n".join(
            part.text
            for message in request.messages
            for part in message.content
            if isinstance(part, TextPart)
        )
        if "recover after" not in request_text:
            async with self._direct_started:
                self.direct_requests.append(request)
                self._direct_started.notify_all()
            await asyncio.wait_for(self._direct_releases.get(), timeout=30)
            self.direct_completions += 1
            yield ModelStreamEvent.text_delta("dashboard session mutation completed")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})
            return

        self.recovery_requests.append(request)
        marker = await asyncio.wait_for(self._replay_releases.get(), timeout=10)
        self.replay_markers.append(marker)
        yield ModelStreamEvent.text_delta("dashboard mutation recovery completed")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class DashboardContractTool(Tool):
    spec = ToolSpec(
        name="dashboard_contract_tool",
        description="Exercise dashboard approval resolution without external side effects.",
        input_schema={
            "type": "object",
            "properties": {"operation": {"type": "string"}},
            "required": ["operation"],
        },
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(
            content=f"Dashboard contract operation {args['operation']} completed.",
            structured={"agent": ctx.agent_name},
        )


class MutationDisconnectFaults:
    """Close two initial run observers without cancelling detached execution."""

    def __init__(self, app: ASGIApp, provider: DashboardContractProvider) -> None:
        self.app = app
        self.provider = provider
        self.initial_run_requests = 0
        self.initial_mutation_ids: list[str] = []
        self.initial_mutation_requests: dict[str, int] = {}
        self.replay_requests: list[tuple[str, str]] = []

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        headers = dict(scope.get("headers", []))
        is_run_post = (
            scope["type"] == "http"
            and scope.get("method") == "POST"
            and scope.get("path") == "/api/run"
        )
        mutation_id = headers.get(b"cayu-mutation-id", b"").decode("ascii", errors="replace")
        replay_header = headers.get(b"last-event-id")
        if (
            scope["type"] == "http"
            and scope.get("method") == "POST"
            and replay_header is None
            and mutation_id
        ):
            path = str(scope.get("path", ""))
            self.initial_mutation_requests[path] = self.initial_mutation_requests.get(path, 0) + 1
        if is_run_post and replay_header is not None:
            replay_marker = replay_header.decode("ascii", errors="replace")
            self.replay_requests.append((replay_marker, mutation_id))
            self.provider.release_after_replay(replay_marker)
            await self.app(scope, receive, send)
            return

        inject = is_run_post and replay_header is None and self.initial_run_requests < 2
        if not inject:
            await self.app(scope, receive, send)
            return

        fault_index = self.initial_run_requests
        self.initial_run_requests += 1
        self.initial_mutation_ids.append(mutation_id)
        response_finished = False

        async def fault_send(message: AsgiMessage) -> None:
            nonlocal response_finished
            if response_finished:
                return
            if message["type"] != "http.response.body":
                await send(message)
                return

            body = message.get("body", b"")
            more_body = message.get("more_body", False)
            if fault_index == 0:
                # Preserve HTTP acceptance but close before the first SSE frame.
                if body or not more_body:
                    response_finished = True
                    # A comment flushes the response headers without creating an
                    # SSE event. An empty final ASGI body can be coalesced with
                    # the headers by the HTTP server, which would model a request
                    # that never opened rather than an accepted observer close.
                    await send(
                        {
                            "type": "http.response.body",
                            "body": b": injected observer close\n\n",
                            "more_body": False,
                        }
                    )
                return

            # Deliver one durable SSE response chunk, then close the observer.
            # The client must reconnect from the exact durable marker established
            # by that frame or its bounded REST reconciliation.
            if body:
                response_finished = True
                await send({"type": "http.response.body", "body": body, "more_body": False})
            elif not more_body:
                response_finished = True
                await send(message)

        await self.app(scope, receive, fault_send)


async def main() -> None:
    app, provider, store = await _seed_app()
    server_app = MutationDisconnectFaults(create_server(app, dev=True), provider)
    listener = _loopback_listener()
    port = listener.getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"
    server = uvicorn.Server(
        uvicorn.Config(
            server_app,
            log_level="warning",
            lifespan="off",
        )
    )
    server_task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        await _wait_for_server(server, server_task)
        evidence = await _run_browser_contract(base_url, provider)
        require_equal(
            server_app.initial_run_requests,
            2,
            "browser recovery must inject both initial observer disconnects",
        )
        require_equal(
            len(provider.recovery_requests),
            2,
            "browser recovery must execute each dashboard run exactly once",
        )
        require_equal(
            len(provider.direct_requests),
            3,
            "resume, approval, and interrupted-resume browser flows must each execute once",
        )
        require_equal(
            provider.direct_completions,
            2,
            "only the explicitly released resume and approval provider calls may complete",
        )
        require_equal(
            server_app.initial_mutation_requests,
            {
                "/api/run": 2,
                "/api/resume": 2,
                "/api/tool-approvals/resolve": 1,
                f"/api/sessions/{INTERRUPT_SESSION_ID}/interrupt": 1,
                f"/api/sessions/{RESUME_INTERRUPT_SESSION_ID}/interrupt": 1,
            },
            "browser session mutations must each be submitted exactly once",
        )
        require_equal(
            len(server_app.replay_requests),
            2,
            "each injected observer close must produce one explicit replay request",
        )
        replay_markers = [marker for marker, _mutation_id in server_app.replay_requests]
        replay_mutation_ids = [mutation_id for _marker, mutation_id in server_app.replay_requests]
        require_equal(
            replay_mutation_ids,
            server_app.initial_mutation_ids,
            "replay requests must preserve the exact submitted mutation identities",
        )
        require_equal(
            provider.replay_markers,
            replay_markers,
            "provider execution must remain blocked until its browser replay is observed",
        )
        require_equal(
            len(set(replay_mutation_ids)),
            2,
            "each dashboard run must use a distinct mutation identity",
        )
        for marker in replay_markers:
            marker_parts = marker.split(":", maxsplit=1)
            require_equal(len(marker_parts), 2, "Last-Event-ID must contain a session and event id")
            session_id, event_id = marker_parts
            require(bool(session_id and event_id), "Last-Event-ID must name a durable event")
            records = await store.query_events(
                EventQuery(session_id=session_id, event_id=event_id, limit=1)
            )
            require_equal(
                len(records),
                1,
                "Last-Event-ID must be the exact identity of an existing durable event",
            )
        evidence["mutation_provider_requests"] = len(provider.requests)
        evidence["injected_initial_disconnects"] = server_app.initial_run_requests
        evidence["mutation_replay_requests"] = len(server_app.replay_requests)
        print(f"{EVIDENCE_PREFIX}{json.dumps(evidence, sort_keys=True)}")
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(server_task, timeout=10)
        except TimeoutError:
            server_task.cancel()
            await asyncio.gather(server_task, return_exceptions=True)


async def _seed_app() -> tuple[CayuApp, DashboardContractProvider, InMemorySessionStore]:
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)
    provider = DashboardContractProvider()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model=MODEL_NAME))
    app.register_agent(
        AgentSpec(name=AGENT_NAME, model=MODEL_NAME),
        tools=[DashboardContractTool()],
    )
    await store.create(
        RunRequest(
            agent_name=AGENT_NAME,
            session_id=SESSION_ID,
            messages=[Message.text("user", "Show the dashboard contract session.")],
        ),
        identity=SessionIdentity(provider_name=PROVIDER_NAME, model=MODEL_NAME),
    )
    await store.append_events(
        SESSION_ID,
        [
            Event(
                id="dashboard-session-started",
                type=EventType.SESSION_STARTED,
                session_id=SESSION_ID,
                agent_name=AGENT_NAME,
            ),
            Event(
                id="dashboard-model-started",
                type=EventType.MODEL_STARTED,
                session_id=SESSION_ID,
                agent_name=AGENT_NAME,
                payload={"provider_name": PROVIDER_NAME, "model": MODEL_NAME},
            ),
            Event(
                id="dashboard-model-completed",
                type=EventType.MODEL_COMPLETED,
                session_id=SESSION_ID,
                agent_name=AGENT_NAME,
                payload={
                    "contract_marker": PAYLOAD_MARKER,
                    "usage_metrics": {
                        "provider_name": PROVIDER_NAME,
                        "requested_model": MODEL_NAME,
                        "model": MODEL_NAME,
                        "input_tokens": 12,
                        "output_tokens": 3,
                        "total_tokens": 15,
                    },
                },
            ),
            Event(
                id="dashboard-tool-failed",
                type=EventType.TOOL_CALL_FAILED,
                session_id=SESSION_ID,
                agent_name=AGENT_NAME,
                tool_name="browser_contract_tool",
                payload={"error": "dashboard contract tool failure"},
            ),
            Event(
                id="dashboard-session-completed",
                type=EventType.SESSION_COMPLETED,
                session_id=SESSION_ID,
                agent_name=AGENT_NAME,
            ),
        ],
    )
    await store.append_transcript_messages(
        SESSION_ID,
        [
            Message.text("user", "dashboard transcript user marker"),
            Message(
                role=MessageRole.ASSISTANT,
                content=(ThinkingPart(text="dashboard transcript thinking marker"),),
            ),
            Message.text("assistant", "dashboard transcript assistant marker"),
        ],
    )
    await store.update_status(SESSION_ID, SessionStatus.COMPLETED)

    async def seed_completed_session(session_id: str, prompt: str) -> None:
        await store.create(
            RunRequest(
                agent_name=AGENT_NAME,
                session_id=session_id,
                messages=[Message.text("user", prompt)],
            ),
            identity=SessionIdentity(provider_name=PROVIDER_NAME, model=MODEL_NAME),
        )
        await store.append_transcript_messages(session_id, [Message.text("user", prompt)])
        await store.append_events(
            session_id,
            [
                Event(
                    id=f"{session_id}-completed",
                    type=EventType.SESSION_COMPLETED,
                    session_id=session_id,
                    agent_name=AGENT_NAME,
                )
            ],
        )
        await store.update_status(session_id, SessionStatus.COMPLETED)

    await seed_completed_session(
        RESUME_INTERRUPT_SESSION_ID,
        "Resume this session, then interrupt its active dashboard observation.",
    )
    await seed_completed_session(
        REOBSERVE_SESSION_ID,
        "Retry an incomplete dashboard mutation observation.",
    )

    for session_id in (INTERRUPT_SESSION_ID, INTERRUPT_FAILURE_SESSION_ID):
        await store.create(
            RunRequest(
                agent_name=AGENT_NAME,
                session_id=session_id,
                messages=[Message.text("user", "Wait for a dashboard interruption.")],
            ),
            identity=SessionIdentity(provider_name=PROVIDER_NAME, model=MODEL_NAME),
        )

    await store.create(
        RunRequest(
            agent_name=AGENT_NAME,
            session_id=APPROVAL_SESSION_ID,
            messages=[Message.text("user", "Resolve the dashboard approval contract.")],
        ),
        identity=SessionIdentity(provider_name=PROVIDER_NAME, model=MODEL_NAME),
    )
    await store.append_transcript_messages(
        APPROVAL_SESSION_ID,
        [
            Message.text("user", "Resolve the dashboard approval contract."),
            Message.tool_call(
                tool_call_id="dashboard-approval-call",
                tool_name="dashboard_contract_tool",
                arguments={"operation": "verify"},
            ),
        ],
    )
    await store.append_events(
        APPROVAL_SESSION_ID,
        [
            Event(
                id="dashboard-approval-requested",
                type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
                session_id=APPROVAL_SESSION_ID,
                agent_name=AGENT_NAME,
                tool_name="dashboard_contract_tool",
                payload={
                    "approval": {
                        "approval_id": "dashboard-approval",
                        "tool_name": "dashboard_contract_tool",
                        "reason": "browser contract decision",
                        "arguments": {"operation": "verify"},
                    }
                },
            )
        ],
    )
    await store.checkpoint(
        APPROVAL_SESSION_ID,
        {
            "pending_tool_approval": {
                "approval_id": "dashboard-approval",
                "tool_call_id": "dashboard-approval-call",
                "tool_name": "dashboard_contract_tool",
                "arguments": {"operation": "verify"},
                "agent_name": AGENT_NAME,
                "tool_calls": [
                    {
                        "tool_call_id": "dashboard-approval-call",
                        "tool_name": "dashboard_contract_tool",
                        "arguments": {"operation": "verify"},
                        "policy_decision": None,
                        "reason": None,
                        "metadata": {},
                        "active_taint_labels": [],
                    }
                ],
            }
        },
    )
    await store.update_status(APPROVAL_SESSION_ID, SessionStatus.INTERRUPTED)
    return app, provider, store


async def _run_browser_contract(
    base_url: str, provider: DashboardContractProvider
) -> dict[str, object]:
    browser_failures: dict[str, list[str]] = {
        "console_errors": [],
        "page_errors": [],
        "request_failures": [],
        "api_errors": [],
    }
    expected_observer_aborts: list[str] = []
    expected_observer_abort_paths = {
        "/api/run",
        "/api/resume",
        "/api/tool-approvals/resolve",
        f"/api/sessions/{INTERRUPT_SESSION_ID}/interrupt",
        f"/api/sessions/{RESUME_INTERRUPT_SESSION_ID}/interrupt",
    }
    diagnostics_dir = Path(
        os.environ.get(
            "CAYU_DASHBOARD_DIAGNOSTICS_DIR",
            str(Path(tempfile.gettempdir()) / "cayu-dashboard-behavior"),
        )
    )

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1440, "height": 1000})
        await context.grant_permissions(["clipboard-read", "clipboard-write"], origin=base_url)
        await context.tracing.start(screenshots=True, snapshots=True)
        page = await context.new_page()
        _record_browser_failures(
            page,
            browser_failures,
            expected_observer_aborts,
            expected_observer_abort_paths,
        )
        try:
            await _exercise_dashboard(page, base_url, provider)
            _require_no_browser_failures(browser_failures)
            run_observer_aborts = [
                failure
                for failure in expected_observer_aborts
                if re.search(r"POST .*/api/run:", failure)
            ]
            require_equal(
                len(run_observer_aborts),
                2,
                "each recovered browser run must close exactly one completed SSE observer",
            )
        except BaseException:
            await _capture_diagnostics(context, page, diagnostics_dir, browser_failures)
            raise
        else:
            await context.tracing.stop()
        finally:
            await browser.close()

    return {
        "browser": "chromium",
        "base_url": base_url,
        "session_id": SESSION_ID,
        "interactions": [
            "mutation_pre_frame_recovery",
            "mutation_post_frame_recovery",
            "session_resume",
            "approval_resolution",
            "session_interrupt",
            "active_resume_interrupt",
            "interrupt_failure_dismissal",
            "manual_mutation_reobservation",
            "sessions_list",
            "session_detail",
            "event_detail",
            "event_filters",
            "exact_event_lookup",
            "filtered_failure_diagnostics",
            "transcript_filters",
            "history_navigation",
        ],
        "console_errors": 0,
        "page_errors": 0,
        "mutation_observer_aborts": len(expected_observer_aborts),
        "request_failures": 0,
        "api_errors": 0,
    }


async def _exercise_dashboard(
    page: Page, base_url: str, provider: DashboardContractProvider
) -> None:
    await _exercise_mutation_recovery(page, base_url)
    await page.goto(f"{base_url}/cayu/sessions", wait_until="networkidle")
    require((await page.locator("body").inner_text()).strip() != "", "dashboard rendered blank")

    await expect(page.get_by_role("heading", name="Sessions").first).to_be_visible()
    session_link = page.get_by_role("link", name=SESSION_ID)
    await expect(session_link).to_be_visible()
    session_row = page.get_by_role("row").filter(has_text=SESSION_ID)
    await expect(session_row.get_by_text("completed", exact=True)).to_be_visible()
    await session_link.click()

    await expect(page).to_have_url(re.compile(rf"/cayu/sessions/{SESSION_ID}$"))
    await expect(page.get_by_role("heading", name=SESSION_ID)).to_be_visible()
    token_stat = page.get_by_text("Tokens", exact=True).locator("..")
    await expect(token_stat.get_by_text("15", exact=True)).to_be_visible()

    completed_event = page.get_by_role("button", name=re.compile(r"model\.completed"))
    await expect(completed_event).to_be_visible()
    await completed_event.click()

    await expect(page.get_by_text("Event Detail", exact=True)).to_be_visible()
    await expect(page.get_by_text("model.completed", exact=True).last).to_be_visible()
    await expect(page.locator("pre").filter(has_text=PAYLOAD_MARKER)).to_be_visible()

    event_type_filter = page.get_by_label("Filter events by exact event type")
    await event_type_filter.fill("model.completed")
    await page.get_by_role("button", name="Apply filters").click()
    await expect(page).to_have_url(re.compile(r"[?&]event_type=model\.completed(?:&|$)"))
    await expect(page.get_by_role("button", name=re.compile(r"model\.completed"))).to_be_visible()
    await expect(page.get_by_role("button", name=re.compile(r"model\.started"))).to_have_count(0)
    await expect(page.get_by_text("Tool failed: browser_contract_tool", exact=True)).to_be_visible()
    await page.get_by_role("button", name="Inspect event").click()
    await expect(page).to_have_url(re.compile(r"[?&]event_id=dashboard-tool-failed(?:&|$)"))
    await expect(page.get_by_text("dashboard-tool-failed", exact=True)).to_be_visible()

    event_id_filter = page.get_by_label("Filter events by exact event ID")
    await event_id_filter.fill("dashboard-model-completed")
    await event_type_filter.fill("model.completed")
    await page.get_by_role("button", name="Apply filters").click()
    await expect(page).to_have_url(re.compile(r"[?&]event_id=dashboard-model-completed(?:&|$)"))
    await expect(page.get_by_text("dashboard-model-completed", exact=True)).to_be_visible()

    transcript_role_filter = page.get_by_label("Filter transcript by role")
    await transcript_role_filter.select_option("assistant")
    await expect(page).to_have_url(re.compile(r"[?&]transcript_role=assistant(?:&|$)"))
    await page.reload(wait_until="networkidle")
    await expect(event_type_filter).to_have_value("model.completed")
    await expect(event_id_filter).to_have_value("dashboard-model-completed")
    await expect(transcript_role_filter).to_have_value("assistant")
    await expect(
        page.get_by_text("dashboard transcript assistant marker", exact=True)
    ).to_be_visible()
    await expect(page.get_by_text("dashboard transcript user marker", exact=True)).to_have_count(0)
    thinking_payload = page.locator("pre").filter(has_text="dashboard transcript thinking marker")
    await expect(thinking_payload).to_be_visible()

    include_thinking = page.get_by_label("Include thinking")
    await include_thinking.uncheck()
    await expect(page).to_have_url(re.compile(r"[?&]include_thinking=false(?:&|$)"))
    await expect(thinking_payload).to_have_count(0)
    await expect(
        page.get_by_text("dashboard transcript assistant marker", exact=True)
    ).to_be_visible()

    await page.go_back()
    await expect(include_thinking).to_be_checked()
    await expect(thinking_payload).to_be_visible()
    await _exercise_existing_session_mutations(page, base_url, provider)


async def _exercise_mutation_recovery(page: Page, base_url: str) -> None:
    session_urls: list[str] = []

    for prompt in (
        "recover after HTTP acceptance before the first event",
        "recover after the first durable event",
    ):
        await page.goto(f"{base_url}/cayu/run", wait_until="networkidle")
        await expect(page.get_by_role("heading", name="New Run")).to_be_visible()
        await page.get_by_placeholder(re.compile(r"Analyze the customer dataset")).fill(prompt)
        await page.get_by_role("button", name="Run", exact=True).click()
        await expect(page.locator('[data-mutation-transport-phase="terminal"]')).to_be_visible(
            timeout=15_000
        )
        await expect(
            page.get_by_text("dashboard mutation recovery completed", exact=True)
        ).to_be_visible()
        session_reference = page.get_by_test_id("run-session-reference")
        await expect(session_reference).to_be_visible()
        session_id = await session_reference.inner_text()
        copy_button = page.get_by_role("button", name="Copy", exact=True)
        await expect(copy_button).to_be_visible()
        await copy_button.click()
        await expect(page.get_by_role("button", name="Copied", exact=True)).to_be_visible()
        session_button = page.get_by_role("button", name="View Session →")
        await expect(session_button).to_be_visible()
        await session_button.click()
        await expect(page).to_have_url(f"{base_url}/cayu/sessions/{session_id}")
        await expect(page.get_by_role("heading", name=session_id)).to_be_visible()
        session_urls.append(page.url)

    require_equal(
        len(set(session_urls)), 2, "each recovered run must keep a distinct session identity"
    )


async def _exercise_existing_session_mutations(
    page: Page, base_url: str, provider: DashboardContractProvider
) -> None:
    await page.goto(f"{base_url}/cayu/sessions/{SESSION_ID}", wait_until="networkidle")
    resume_input = page.get_by_placeholder("Continue with a new prompt...")
    await expect(resume_input).to_be_visible()
    await resume_input.fill("Resume through the dashboard browser contract.")
    resume_button = page.get_by_role("button", name="Resume", exact=True)
    await resume_button.click()
    await provider.wait_for_direct_requests(1)
    await expect(page.get_by_role("button", name="Interrupt session", exact=True)).to_be_visible(
        timeout=10_000
    )
    await expect(resume_button).to_have_count(0)
    provider.release_direct()
    await expect(page.locator('[data-mutation-transport-phase="terminal"]')).to_be_visible(
        timeout=15_000
    )
    await expect(page.get_by_role("button", name=re.compile(r"session\.resumed"))).to_be_visible()

    await page.goto(f"{base_url}/cayu/sessions/{APPROVAL_SESSION_ID}", wait_until="networkidle")
    await expect(page.get_by_text("Awaiting approval", exact=True)).to_be_visible()
    deny_button = page.get_by_role("button", name="Deny", exact=True)
    await deny_button.click()
    await provider.wait_for_direct_requests(2)
    await expect(page.get_by_role("button", name="Interrupt session", exact=True)).to_be_visible(
        timeout=10_000
    )
    await expect(deny_button).to_have_count(0)
    provider.release_direct()
    await expect(page.locator('[data-mutation-transport-phase="terminal"]')).to_be_visible(
        timeout=15_000
    )
    await expect(page.get_by_text("Awaiting approval", exact=True)).to_have_count(0)

    await page.goto(f"{base_url}/cayu/sessions/{INTERRUPT_SESSION_ID}", wait_until="networkidle")
    await page.get_by_role("button", name="Interrupt session", exact=True).click()
    interrupt_sheet = page.get_by_role("dialog")
    await expect(interrupt_sheet.get_by_text("Interrupt session?", exact=True)).to_be_visible()
    await expect(interrupt_sheet.get_by_role("button", name="Keep running")).to_be_visible()
    await interrupt_sheet.get_by_label("Reason (optional)").fill("browser contract interruption")
    await interrupt_sheet.get_by_role("button", name="Interrupt session", exact=True).click()
    await expect(page.locator('[data-mutation-transport-phase="terminal"]')).to_be_visible(
        timeout=15_000
    )
    await expect(interrupt_sheet).to_be_hidden()
    await expect(page.get_by_text("interrupted", exact=True).first).to_be_visible()

    await page.goto(
        f"{base_url}/cayu/sessions/{RESUME_INTERRUPT_SESSION_ID}",
        wait_until="networkidle",
    )
    interrupted_resume_input = page.get_by_placeholder("Continue with a new prompt...")
    await interrupted_resume_input.fill("Interrupt this resume while its provider is active.")
    interrupted_resume_button = page.get_by_role("button", name="Resume", exact=True)
    await interrupted_resume_button.click()
    await provider.wait_for_direct_requests(3)
    active_interrupt_button = page.get_by_role("button", name="Interrupt session", exact=True)
    await expect(active_interrupt_button).to_be_visible(timeout=10_000)
    await expect(interrupted_resume_button).to_have_count(0)
    await active_interrupt_button.click()
    active_interrupt_sheet = page.get_by_role("dialog")
    await active_interrupt_sheet.get_by_label("Reason (optional)").fill(
        "interrupt an active resumed run"
    )
    await active_interrupt_sheet.get_by_role("button", name="Interrupt session", exact=True).click()
    await expect(page.locator('[data-mutation-transport-phase="terminal"]')).to_be_visible(
        timeout=15_000
    )
    await expect(active_interrupt_sheet).to_be_hidden()
    await expect(page.get_by_text("interrupted", exact=True).first).to_be_visible()

    failure_path = f"**/api/sessions/{INTERRUPT_FAILURE_SESSION_ID}/interrupt"

    async def inject_runtime_failure(route) -> None:
        error = {
            "type": "stream.error",
            "kind": "runtime",
            "code": "runtime_failed",
            "error": "Injected dashboard interrupt failure.",
            "error_type": "DashboardContractFailure",
            "retryable": False,
            "session_id": INTERRUPT_FAILURE_SESSION_ID,
        }
        await route.fulfill(
            status=200,
            headers={"content-type": "text/event-stream"},
            body=f"event: error\ndata: {json.dumps(error, separators=(',', ':'))}\n\n",
        )

    await page.route(failure_path, inject_runtime_failure)
    try:
        await page.goto(
            f"{base_url}/cayu/sessions/{INTERRUPT_FAILURE_SESSION_ID}",
            wait_until="networkidle",
        )
        await page.get_by_role("button", name="Interrupt session", exact=True).click()
        failure_sheet = page.get_by_role("dialog")
        await failure_sheet.get_by_role("button", name="Interrupt session", exact=True).click()
        await expect(
            page.locator('[data-mutation-transport-phase="runtime_failed"]')
        ).to_be_visible(timeout=15_000)
        close_action = failure_sheet.get_by_role("button", name="Close", exact=True).first
        await expect(close_action).to_be_visible()
        await expect(failure_sheet.get_by_role("button", name="Keep running")).to_have_count(0)
        await close_action.click()
        await expect(failure_sheet).to_be_hidden()
    finally:
        await page.unroute(failure_path, inject_runtime_failure)

    await _exercise_manual_mutation_reobservation(page, base_url)


async def _exercise_manual_mutation_reobservation(page: Page, base_url: str) -> None:
    resume_path = "**/api/resume"
    events_path = f"**/api/sessions/{REOBSERVE_SESSION_ID}/events*"
    baseline_response = await page.request.get(
        f"{base_url}/api/sessions/{REOBSERVE_SESSION_ID}/events",
        params={"event_id": f"{REOBSERVE_SESSION_ID}-completed", "limit": 1},
    )
    require_equal(baseline_response.status, 200, "manual recovery baseline must be readable")
    baseline_payload = await baseline_response.json()
    baseline_events = baseline_payload.get("events", [])
    require_equal(len(baseline_events), 1, "manual recovery requires one durable baseline event")
    baseline_sequence = baseline_events[0]["sequence"]
    require(
        isinstance(baseline_sequence, int) and baseline_sequence > 0,
        "manual recovery baseline must have a positive durable sequence",
    )
    mutation_id: str | None = None
    recovered = False
    timestamp = "2026-01-01T00:00:00Z"
    terminal_event = {
        "id": f"{REOBSERVE_SESSION_ID}-recovered-completed",
        "type": "session.completed",
        "session_id": REOBSERVE_SESSION_ID,
        "timestamp": timestamp,
        "agent_name": AGENT_NAME,
        "tool_name": None,
        "environment_name": None,
        "workflow_name": None,
        "payload": {},
    }

    async def inject_observer_failure_then_recovery(route) -> None:
        nonlocal mutation_id, recovered
        request_headers = route.request.headers
        request_mutation_id = request_headers.get("cayu-mutation-id")
        require(bool(request_mutation_id), "dashboard resume must carry a mutation identity")
        last_event_id = request_headers.get("last-event-id")
        if last_event_id is None:
            require(mutation_id is None, "manual recovery may submit the mutation only once")
            mutation_id = request_mutation_id
            error = {
                "type": "stream.error",
                "kind": "observer",
                "code": "event_frame_too_large",
                "error": "Injected non-retryable dashboard observer failure.",
                "error_type": "DashboardContractObserverFailure",
                "retryable": False,
                "session_id": REOBSERVE_SESSION_ID,
            }
            await route.fulfill(
                status=200,
                headers={"content-type": "text/event-stream"},
                body=f"event: error\ndata: {json.dumps(error, separators=(',', ':'))}\n\n",
            )
            return

        require_equal(
            request_mutation_id,
            mutation_id,
            "manual recovery must preserve the original mutation identity",
        )
        require_equal(
            last_event_id,
            f"{REOBSERVE_SESSION_ID}:{REOBSERVE_SESSION_ID}-completed",
            "manual recovery must replay from the original durable baseline",
        )
        acceptance_event = {
            "id": f"{REOBSERVE_SESSION_ID}-mutation-accepted",
            "type": "server.mutation.accepted",
            "session_id": REOBSERVE_SESSION_ID,
            "timestamp": timestamp,
            "agent_name": AGENT_NAME,
            "tool_name": None,
            "environment_name": None,
            "workflow_name": None,
            "payload": {
                "mutation_id": mutation_id,
                "mutation_kind": "resume",
                "accepted_event_id": terminal_event["id"],
                "accepted_event_type": terminal_event["type"],
            },
        }
        recovered = True
        frames = []
        for event in (terminal_event, acceptance_event):
            frames.append(
                f"id: {REOBSERVE_SESSION_ID}:{event['id']}\n"
                f"data: {json.dumps(event, separators=(',', ':'))}\n\n"
            )
        await route.fulfill(
            status=200,
            headers={"content-type": "text/event-stream"},
            body="".join(frames),
        )

    async def expose_recovered_events(route) -> None:
        if not recovered:
            await route.continue_()
            return

        query = parse_qs(urlsplit(route.request.url).query)
        order_by = query.get("order_by", ["sequence_asc"])[0]
        limit = int(query.get("limit", ["100"])[0])
        after_sequence = int(query.get("after_sequence", ["0"])[0])
        before_value = query.get("before_sequence", [None])[0]
        before_sequence = int(before_value) if before_value is not None else None
        records = [
            {**terminal_event, "sequence": baseline_sequence + 1},
            {
                "id": f"{REOBSERVE_SESSION_ID}-mutation-accepted",
                "type": "server.mutation.accepted",
                "session_id": REOBSERVE_SESSION_ID,
                "timestamp": timestamp,
                "agent_name": AGENT_NAME,
                "tool_name": None,
                "environment_name": None,
                "workflow_name": None,
                "payload": {
                    "mutation_id": mutation_id,
                    "mutation_kind": "resume",
                    "accepted_event_id": terminal_event["id"],
                    "accepted_event_type": terminal_event["type"],
                },
                "sequence": baseline_sequence + 2,
            },
        ]
        matching = [record for record in records if record["sequence"] > after_sequence]
        if before_sequence is not None:
            matching = [record for record in matching if record["sequence"] < before_sequence]
        if order_by == "sequence_desc":
            matching.reverse()
        page_records = matching[:limit]
        payload = {
            "session_id": REOBSERVE_SESSION_ID,
            "events": page_records,
            "order_by": order_by,
            "next_sequence": page_records[-1]["sequence"] if page_records else None,
            "scan_through_sequence": baseline_sequence + 2 if before_sequence is None else None,
            "has_more": len(matching) > limit,
        }
        await route.fulfill(
            status=200,
            headers={"content-type": "application/json"},
            body=json.dumps(payload, separators=(",", ":")),
        )

    await page.route(resume_path, inject_observer_failure_then_recovery)
    await page.route(events_path, expose_recovered_events)
    try:
        await page.goto(
            f"{base_url}/cayu/sessions/{REOBSERVE_SESSION_ID}", wait_until="networkidle"
        )
        resume_input = page.get_by_placeholder("Continue with a new prompt...")
        await resume_input.fill("Recover this dashboard observation without resubmitting.")
        await page.get_by_role("button", name="Resume", exact=True).click()
        await expect(
            page.locator('[data-mutation-transport-phase="transport_failed"]')
        ).to_be_visible(timeout=25_000)
        await expect(resume_input).to_have_count(0)
        retry_observation = page.get_by_role("button", name="Retry observation", exact=True)
        await expect(retry_observation).to_be_visible()
        await retry_observation.click()
        await expect(page.locator('[data-mutation-transport-phase="terminal"]')).to_be_visible(
            timeout=15_000
        )
        await expect(retry_observation).to_have_count(0)
        await expect(resume_input).to_be_enabled()
        require(recovered, "manual dashboard recovery must issue a replay request")
    finally:
        await page.unroute(events_path, expose_recovered_events)
        await page.unroute(resume_path, inject_observer_failure_then_recovery)


def _record_browser_failures(
    page: Page,
    failures: dict[str, list[str]],
    expected_observer_aborts: list[str],
    expected_observer_abort_paths: set[str],
) -> None:
    def record_request_failure(request: Request) -> None:
        detail = f"{request.method} {request.url}: {request.failure or 'unknown failure'}"
        # The client deliberately aborts only these mutation observers after a
        # durable terminal boundary or when an explicit interrupt supersedes an
        # active observation. Every other failed API request remains a failure.
        if (
            request.method == "POST"
            and urlsplit(request.url).path in expected_observer_abort_paths
            and request.failure == "net::ERR_ABORTED"
        ):
            expected_observer_aborts.append(detail)
            return
        failures["request_failures"].append(detail)

    page.on(
        "console",
        lambda message: (
            failures["console_errors"].append(message.text) if message.type == "error" else None
        ),
    )
    page.on("pageerror", lambda error: failures["page_errors"].append(str(error)))
    page.on("requestfailed", record_request_failure)
    page.on(
        "response",
        lambda response: (
            failures["api_errors"].append(f"{response.status} {response.url}")
            if "/api/" in response.url and response.status >= 400
            else None
        ),
    )


def _require_no_browser_failures(failures: dict[str, list[str]]) -> None:
    for kind, messages in failures.items():
        require(not messages, f"dashboard recorded {kind}: {messages}")


async def _capture_diagnostics(
    context: BrowserContext,
    page: Page,
    diagnostics_dir: Path,
    failures: dict[str, list[str]],
) -> None:
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    screenshot = diagnostics_dir / "dashboard-behavior.png"
    trace = diagnostics_dir / "dashboard-behavior-trace.zip"
    html = diagnostics_dir / "dashboard-behavior.html"
    capture_errors: list[str] = []
    try:
        await page.screenshot(path=str(screenshot), full_page=True)
    except Exception as exc:
        capture_errors.append(f"screenshot: {exc}")
    try:
        html.write_text(await page.content(), encoding="utf-8")
    except Exception as exc:
        capture_errors.append(f"html: {exc}")
    try:
        await context.tracing.stop(path=str(trace))
    except Exception as exc:
        capture_errors.append(f"trace: {exc}")
    print(
        "CAYU_DASHBOARD_DIAGNOSTICS="
        + json.dumps(
            {
                "directory": str(diagnostics_dir),
                "browser_failures": failures,
                "capture_errors": capture_errors,
            },
            sort_keys=True,
        )
    )


def _loopback_listener() -> socket.socket:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.setblocking(False)
    return listener


async def _wait_for_server(server: uvicorn.Server, task: asyncio.Task[None]) -> None:
    async def wait_until_started() -> None:
        while not server.started:
            if task.done():
                await task
                raise RuntimeError("dashboard server stopped before startup")
            await asyncio.sleep(0.01)

    await asyncio.wait_for(wait_until_started(), timeout=10)


if __name__ == "__main__":
    asyncio.run(main())
