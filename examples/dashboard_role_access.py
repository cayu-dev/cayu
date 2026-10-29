"""Developer dashboard roles against a real server.

`dashboard_behavior_live.py` checks the member pages by rewriting `/api/app` in the
browser, so the server behind them still grants full access. This part serves the
packaged dashboard behind proxy identity with the default `ControlPlaneRoles`. Every
browser request carries a freshly signed identity, so the server decides each request
by the viewer's real role. It checks a developer and a member across the shell, Home,
a session, Inbox, Tasks, Specialists, Tools, Knowledge, Evals, Usage, Environments,
Artifacts and System, including
the 403 `role_forbidden` a member gets when a write reaches the server anyway.

Run it alone with `CAYU_DASHBOARD_PART=roles python examples/dashboard_behavior_live.py`.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import socket
import tempfile
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import jwt
import uvicorn
from cryptography.hazmat.primitives.asymmetric import ec
from jwt.algorithms import ECAlgorithm
from playwright.async_api import Browser, BrowserContext, Page, Route, async_playwright, expect
from sse_starlette.sse import AppStatus

from _live_checks import require, require_equal
from cayu import (
    AgentSpec,
    CayuApp,
    Message,
    RunRequest,
    Tool,
    ToolContext,
    ToolEffect,
    ToolResult,
    ToolSpec,
)
from cayu.evals import CorpusTarget
from cayu.messages import TextPart, ToolResultPart
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.server import BasicAuth, EvalsConfig, ProxyIdentityAuth, ServerConfig, create_server
from cayu.server.proxy_identity import DEFAULT_PROXY_IDENTITY_HEADER, PROXY_IDENTITY_TOKEN_TYPE
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionStatus
from cayu.storage import SQLiteEvalStore
from cayu.storage.memory import (
    InMemoryKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeEntry,
    KnowledgeStatus,
)
from cayu.tasks import InMemoryTaskStore, TaskCreate, TaskStatus, TaskTypeSpec
from cayu.tools import AlwaysRequireApprovalToolPolicy, UserInputTool

AGENT_NAME = "dashboard-roles-agent"
MODEL_NAME = "dashboard-roles-model"
TOOL_NAME = "dashboard_roles_tool"
TASK_TYPE = "dashboard-roles.review"
ISSUER = "https://sign-in.dashboard-roles.test"
JWKS_URL = f"{ISSUER}/jwks"
AUDIENCE = "agent:dashboard-roles"
SIGNING_KEY_ID = "dashboard-roles-key"
IDENTITY_HEADER = DEFAULT_PROXY_IDENTITY_HEADER.lower()
OPERATOR = ("dashboard-roles-operator", "dashboard-roles-password")
MEMBER_APPROVAL_SESSION = "dashboard-roles-member-approval"
MEMBER_DENIAL_SESSION = "dashboard-roles-member-denial"
MEMBER_QUESTION_SESSION = "dashboard-roles-member-question"
DEVELOPER_APPROVAL_SESSION = "dashboard-roles-developer-approval"
MEMBER_TASK = "dashboard-roles-task-member"
DEVELOPER_TASK = "dashboard-roles-task-developer"
MEMBER_PROPOSAL = "dashboard-roles-proposal-member"
DEVELOPER_PROPOSAL = "dashboard-roles-proposal-developer"
MEMBER_KEPT = "dashboard-roles-kept-member"
DEVELOPER_KEPT = "dashboard-roles-kept-developer"
ROLE_FORBIDDEN_REASON = "Your role can read and decide, but this action requires full access."
ORIGINAL_ARGUMENTS = {"operation": "original"}
CHANGED_ARGUMENTS = {"operation": "changed-by-developer"}
FORCED_ARGUMENTS = {"operation": "changed-by-member"}


class RoleAccessProvider(ModelProvider):
    """Asks for an approval or a question when the prompt says so, then finishes."""

    name = "dashboard-roles-provider"

    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.calls += 1
        if any(
            isinstance(part, ToolResultPart)
            for message in request.messages
            for part in message.content
        ):
            yield ModelStreamEvent.text_delta("Role check finished.")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})
            return
        prompt = "\n".join(
            part.text
            for message in request.messages
            for part in message.content
            if isinstance(part, TextPart)
        ).lower()
        if "approval" in prompt:
            yield ModelStreamEvent.tool_call(
                id=f"dashboard-roles-approval-{self.calls}",
                name=TOOL_NAME,
                arguments=ORIGINAL_ARGUMENTS,
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        if "question" in prompt:
            yield ModelStreamEvent.tool_call(
                id=f"dashboard-roles-question-{self.calls}",
                name="ask_user",
                arguments={
                    "question": "Which report should the agent send?",
                    "options": ["weekly", "monthly"],
                },
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        yield ModelStreamEvent.text_delta("Developer session finished.")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class RoleAccessTool(Tool):
    """Records the arguments each approved call ran with; it has no side effects."""

    spec = ToolSpec(
        name=TOOL_NAME,
        description="Record an approved dashboard role check.",
        input_schema={
            "type": "object",
            "properties": {"operation": {"type": "string"}},
            "required": ["operation"],
        },
        effect=ToolEffect.NONE,
        max_terminal_payload_bytes=1024,
    )

    def __init__(self) -> None:
        self.runs: list[tuple[str, dict]] = []

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.runs.append((ctx.session_id, dict(args)))
        return ToolResult(content=f"Recorded {args['operation']}.")


class IdentitySigner:
    """Stands in for the sign-in proxy: signs short-lived identities with one ES256 key."""

    def __init__(self) -> None:
        self._key = ec.generate_private_key(ec.SECP256R1())
        jwk = ECAlgorithm.to_jwk(self._key.public_key(), as_dict=True)
        self._jwks = {"keys": [{**jwk, "kid": SIGNING_KEY_ID, "use": "sig", "alg": "ES256"}]}

    def http_client(self) -> httpx.AsyncClient:
        def serve(request: httpx.Request) -> httpx.Response:
            if str(request.url) == JWKS_URL:
                return httpx.Response(200, json=self._jwks)
            return httpx.Response(404)

        return httpx.AsyncClient(transport=httpx.MockTransport(serve))

    def token(self, role: str) -> str:
        now = int(time.time())
        claims = {
            "iss": ISSUER,
            "aud": AUDIENCE,
            "sub": f"dashboard-{role}",
            "org": "dashboard-roles-org",
            "role": role,
            "name": f"Dashboard {role}",
            "iat": now,
            "exp": now + 60,
            "jti": uuid.uuid4().hex,
        }
        return jwt.encode(
            claims,
            self._key,
            algorithm="ES256",
            headers={"kid": SIGNING_KEY_ID, "typ": PROXY_IDENTITY_TOKEN_TYPE},
        )


_QUEUE_PROBE = re.compile(r"/api/sessions/[^/]+/messages")


class BrowserFailures:
    """Unexpected browser errors, and the forbidden responses a check expects."""

    def __init__(self) -> None:
        self.unexpected: list[str] = []
        self.forbidden: list[tuple[str, str]] = []
        self.allow_forbidden = False

    def watch(self, page: Page, label: str) -> None:
        page.on("pageerror", lambda error: self.unexpected.append(f"{label} pageerror: {error}"))
        page.on("console", lambda message: self._console(label, message))
        page.on("response", lambda response: self._response(label, response))

    def _console(self, label: str, message) -> None:
        if message.type != "error":
            return
        # Chromium logs every failed resource; forbidden ones are checked as responses.
        forbidden = "status of 403" in message.text
        if forbidden and (
            self.allow_forbidden or _QUEUE_PROBE.fullmatch(urlsplit(message.location["url"]).path)
        ):
            return
        self.unexpected.append(f"{label} console: {message.text} ({message.location['url']})")

    def _response(self, label: str, response) -> None:
        path = urlsplit(response.url).path
        if not path.startswith("/api/") or response.status < 400:
            return
        method = response.request.method
        # The app has no session-message policy, so a full-access page's queue
        # inspection, which probes whether follow-ups can be queued, is refused.
        if response.status == 403 and method == "GET" and _QUEUE_PROBE.fullmatch(path):
            if self.allow_forbidden:
                self.forbidden.append((method, path))
            return
        if response.status == 403 and self.allow_forbidden:
            self.forbidden.append((method, path))
            return
        self.unexpected.append(f"{label} {method} {path}: HTTP {response.status}")

    @asynccontextmanager
    async def forbidden_allowed(self) -> AsyncIterator[list[tuple[str, str]]]:
        start = len(self.forbidden)
        self.allow_forbidden = True
        observed: list[tuple[str, str]] = []
        try:
            yield observed
        finally:
            self.allow_forbidden = False
            observed.extend(self.forbidden[start:])

    def require_none(self) -> None:
        require(
            not self.unexpected, f"dashboard roles recorded browser failures: {self.unexpected}"
        )


async def _task_status(tasks: InMemoryTaskStore, task_id: str) -> TaskStatus:
    task = await tasks.load_task(task_id)
    require(task is not None, f"task {task_id} is missing from the task store")
    assert task is not None
    return task.status


async def exercise_role_access() -> dict[str, object]:
    signer = IdentitySigner()
    app, tool, sessions, tasks = await _seed_app()
    root = tempfile.TemporaryDirectory(prefix="cayu-dashboard-roles-")
    eval_store = SQLiteEvalStore(Path(root.name) / "evals.sqlite")
    auth = ProxyIdentityAuth(
        issuer=ISSUER,
        audience=AUDIENCE,
        jwks_url=JWKS_URL,
        http_client=signer.http_client(),
        fallback=BasicAuth(username=OPERATOR[0], password=OPERATOR[1]),
    )
    server_app = create_server(
        app,
        config=ServerConfig.protected(
            auth,
            evals=EvalsConfig(
                target=CorpusTarget(
                    key="dashboard-roles.regressions",
                    app=app,
                    request_base=RunRequest(agent_name=AGENT_NAME, messages=[]),
                    application_release_id="dashboard-roles",
                ),
                store=eval_store,
            ),
        ),
    )
    failures = BrowserFailures()
    try:
        async with _serve(server_app) as base_url, async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True)
            try:
                developer = await _signed_context(browser, base_url, signer, "developer")
                member = await _signed_context(browser, base_url, signer, "member")
                developer_page = await developer.new_page()
                member_page = await member.new_page()
                failures.watch(developer_page, "developer")
                failures.watch(member_page, "member")
                try:
                    await _exercise_developer(developer_page, base_url, tool, sessions, tasks)
                    forced = await _exercise_member(
                        member_page, base_url, signer, failures, tool, sessions, tasks
                    )
                    failures.require_none()
                except BaseException:
                    await _capture(developer_page, member_page)
                    raise
            finally:
                await browser.close()
    finally:
        await eval_store.close()
        root.cleanup()
    return {
        "browser": "chromium",
        "auth": "proxy-identity",
        "roles": ["developer", "member"],
        "member_forbidden_requests": sorted({f"{method} {path}" for method, path in forced}),
        "interactions": [
            "developer_start_session",
            "developer_approve_with_changed_arguments",
            "developer_create_hold_resume_task",
            "developer_knowledge_drop_and_remove",
            "developer_evals_and_system",
            "member_read_and_decide_shell",
            "member_evals_read_only_and_system_explained",
            "member_forced_start_session_forbidden",
            "member_forced_task_hold_forbidden",
            "member_changed_arguments_forbidden",
            "member_inbox_approve_deny_answer",
            "member_tasks_specialists_tools_knowledge_read",
        ],
    }


async def _seed_app() -> tuple[CayuApp, RoleAccessTool, InMemorySessionStore, InMemoryTaskStore]:
    sessions = InMemorySessionStore()
    tasks = InMemoryTaskStore()
    knowledge_scope = KnowledgeAccessScope.privileged()
    knowledge = InMemoryKnowledgeStore(
        [
            KnowledgeEntry(
                id=MEMBER_KEPT, text="Members read kept knowledge.", status=KnowledgeStatus.ACTIVE
            ),
            KnowledgeEntry(
                id=DEVELOPER_KEPT,
                text="Developers remove kept knowledge.",
                status=KnowledgeStatus.ACTIVE,
            ),
            KnowledgeEntry(
                id=MEMBER_PROPOSAL,
                text="Members keep proposed knowledge.",
                status=KnowledgeStatus.PENDING,
            ),
            KnowledgeEntry(
                id=DEVELOPER_PROPOSAL,
                text="Developers drop proposed knowledge.",
                status=KnowledgeStatus.PENDING,
            ),
        ],
        access_scope=knowledge_scope,
    )
    app = CayuApp(
        session_store=sessions,
        task_store=tasks,
        knowledge_store=knowledge,
        knowledge_access_scope=knowledge_scope,
        enable_logging=False,
    )
    app.register_provider(RoleAccessProvider(), default=True)
    tool = RoleAccessTool()
    app.register_agent(
        AgentSpec(name=AGENT_NAME, model=MODEL_NAME),
        tools=[tool, UserInputTool()],
        tool_policy=AlwaysRequireApprovalToolPolicy(tools=[TOOL_NAME]),
    )
    app.register_task_type(
        TaskTypeSpec(
            name=TASK_TYPE,
            description="Review one topic.",
            agent_name=AGENT_NAME,
            input_schema={
                "type": "object",
                "properties": {"topic": {"type": "string", "description": "What to review"}},
                "required": ["topic"],
            },
        )
    )
    for session_id, prompt in (
        (MEMBER_APPROVAL_SESSION, "Ask for the member approval."),
        (MEMBER_DENIAL_SESSION, "Ask for the member approval to deny."),
        (MEMBER_QUESTION_SESSION, "Ask the member question."),
        (DEVELOPER_APPROVAL_SESSION, "Ask for the developer approval."),
    ):
        async for _ in app.run(
            RunRequest(
                agent_name=AGENT_NAME,
                session_id=session_id,
                messages=[Message.text("user", prompt)],
            )
        ):
            pass
    for task_id, title in (
        (MEMBER_TASK, "Member reads this task"),
        (DEVELOPER_TASK, "Developer holds this task"),
    ):
        await app.create_task(
            TaskCreate(
                task_id=task_id,
                type=TASK_TYPE,
                title=title,
                input={"topic": title},
                assigned_agent_name=AGENT_NAME,
            )
        )
    return app, tool, sessions, tasks


async def _exercise_developer(
    page: Page,
    base_url: str,
    tool: RoleAccessTool,
    sessions: InMemorySessionStore,
    tasks: InMemoryTaskStore,
) -> None:
    await page.goto(f"{base_url}/cayu/", wait_until="networkidle")
    await expect(page.get_by_role("heading", name="Home", exact=True)).to_be_visible()
    await expect(page.get_by_text("Read and decide", exact=True)).to_have_count(0)
    sidebar = page.get_by_role("navigation", name="Main")
    for label in ("Evals", "System", "Tasks", "Specialists", "Tools", "Knowledge"):
        await expect(sidebar.get_by_role("link", name=label, exact=True)).to_be_visible()

    # Starting a session runs it on the server and opens it.
    await page.get_by_role("button", name="Start a session", exact=True).click()
    await page.locator("#start-specialist").select_option(AGENT_NAME)
    await page.locator("#start-prompt").fill("Developer start check.")
    await page.get_by_role("button", name="Start", exact=True).click()
    await expect(page).to_have_url(re.compile(r"/cayu/sessions/[^/?]+$"), timeout=15000)
    await expect(page.get_by_text("Developer session finished.", exact=True)).to_be_visible()
    await expect(page.locator("#session-composer")).to_be_visible()

    # Full access may approve with changed arguments; the tool runs with them.
    await page.goto(
        f"{base_url}/cayu/sessions/{DEVELOPER_APPROVAL_SESSION}", wait_until="networkidle"
    )
    card = page.locator("[data-inbox-item]").first
    await card.get_by_text("Decision options", exact=True).click()
    await card.get_by_label("Approve with changed arguments", exact=True).check()
    await card.get_by_label("Changed arguments", exact=True).fill(json.dumps(CHANGED_ARGUMENTS))
    async with page.expect_response(_settle_response) as settled:
        await card.get_by_role("button", name="Approve", exact=True).click()
    require_equal((await settled.value).status, 200, "developer changed-argument approval")
    await _wait_for_status(sessions, DEVELOPER_APPROVAL_SESSION, SessionStatus.COMPLETED)
    require(
        (DEVELOPER_APPROVAL_SESSION, CHANGED_ARGUMENTS) in tool.runs,
        f"the developer's changed arguments must run the tool: {tool.runs}",
    )

    # Tasks: create one of a declared type, then hold and resume another.
    await page.goto(f"{base_url}/cayu/tasks", wait_until="networkidle")
    await page.get_by_role("button", name="Create a task", exact=True).click()
    await page.locator("#start-field-topic").fill("Developer-created review")
    async with page.expect_response(
        lambda response: (
            response.request.method == "POST" and urlsplit(response.url).path == "/api/tasks"
        )
    ) as created:
        await page.get_by_role("button", name="Create task", exact=True).click()
    require((await created.value).ok, "developer task creation must succeed")
    await expect(page).to_have_url(re.compile(r"/cayu/tasks/[^/?]+$"), timeout=15000)
    await expect(page.get_by_text("Developer-created review", exact=True)).to_be_visible()

    await page.goto(f"{base_url}/cayu/tasks/{DEVELOPER_TASK}", wait_until="networkidle")
    await expect(page.get_by_role("button", name="Cancel task", exact=True)).to_be_visible()
    await page.get_by_role("button", name="More actions", exact=True).click()
    await page.get_by_role("button", name="Pause", exact=True).click()
    await page.get_by_role("button", name="Pause task", exact=True).click()
    await expect(page.get_by_role("button", name="Resume", exact=True)).to_be_visible()
    require_equal(await _task_status(tasks, DEVELOPER_TASK), TaskStatus.PAUSED, "developer hold")
    await page.get_by_role("button", name="Resume", exact=True).click()
    await expect(page.get_by_role("button", name="Resume", exact=True)).to_have_count(0)
    require_equal(await _task_status(tasks, DEVELOPER_TASK), TaskStatus.PENDING, "developer resume")

    # Agent pages: a developer may start sessions, inspect tools and change knowledge.
    await page.goto(f"{base_url}/cayu/specialists", wait_until="networkidle")
    await page.get_by_role("button", name=f"Inspect {AGENT_NAME}", exact=True).click()
    await expect(page.locator("main").get_by_role("link", name="Start a session")).to_be_visible()
    await page.goto(f"{base_url}/cayu/tools", wait_until="networkidle")
    await page.get_by_role("button", name=f"Inspect {TOOL_NAME}", exact=True).click()
    await expect(page.get_by_text("Input schema", exact=True)).to_be_visible()
    await page.goto(f"{base_url}/cayu/knowledge", wait_until="networkidle")
    proposal = page.locator("[data-inbox-item]").filter(
        has_text="Developers drop proposed knowledge."
    )
    async with page.expect_response(_settle_response) as dropped:
        await proposal.get_by_role("button", name="Drop", exact=True).click()
    require_equal((await dropped.value).status, 200, "developer knowledge drop")
    kept = page.get_by_role("article", name=f"Knowledge {DEVELOPER_KEPT}", exact=True)
    await kept.get_by_role("button", name="Remove", exact=True).click()
    await kept.get_by_role("button", name="Confirm remove", exact=True).click()
    await expect(kept).to_have_count(0)

    await page.goto(f"{base_url}/cayu/evals/settings", wait_until="networkidle")
    await expect(
        page.get_by_role("heading", name="What trials run with", exact=True)
    ).to_be_visible()
    await _exercise_remaining_reads(page, base_url)
    await page.goto(f"{base_url}/cayu/system", wait_until="networkidle")
    await expect(page.get_by_text("About this agent", exact=True)).to_be_visible()
    await expect(page.get_by_test_id("dashboard-role-unavailable")).to_have_count(0)
    await expect(page.get_by_role("heading", name="System", exact=True)).to_be_visible()


async def _exercise_remaining_reads(page: Page, base_url: str) -> None:
    await page.goto(f"{base_url}/cayu/usage", wait_until="networkidle")
    await expect(page.get_by_role("button", name="Last 30 days", exact=True)).to_be_visible()
    await expect(page.get_by_text("Sessions per day", exact=True)).to_be_visible()
    await page.goto(f"{base_url}/cayu/environments", wait_until="networkidle")
    await expect(
        page.get_by_text(
            "Where your agent's sessions run, and what they can reach there.", exact=True
        )
    ).to_be_visible()
    # This role fixture has no artifact store. The same read capability gate must
    # explain unavailable storage before mounting a page that would issue reads.
    await page.goto(f"{base_url}/cayu/artifacts", wait_until="networkidle")
    await expect(page.get_by_test_id("dashboard-capability-unavailable")).to_be_visible()


async def _exercise_member(
    page: Page,
    base_url: str,
    signer: IdentitySigner,
    failures: BrowserFailures,
    tool: RoleAccessTool,
    sessions: InMemorySessionStore,
    tasks: InMemoryTaskStore,
) -> list[tuple[str, str]]:
    system_reads: list[str] = []
    page.on(
        "request",
        lambda request: (
            system_reads.append(request.url)
            if urlsplit(request.url).path.startswith("/api/system")
            else None
        ),
    )

    # The shell says what the member can do and leaves out what they can't.
    await page.goto(f"{base_url}/cayu/", wait_until="networkidle")
    await expect(page.get_by_role("heading", name="Home", exact=True)).to_be_visible()
    await expect(page.locator("header").get_by_text("Read and decide", exact=True)).to_be_visible()
    await expect(page.get_by_text(ROLE_FORBIDDEN_REASON, exact=True).first).to_be_visible()
    await expect(page.get_by_role("button", name="Start a session", exact=True)).to_have_count(0)
    sidebar = page.get_by_role("navigation", name="Main")
    for label in ("Home", "Inbox", "Sessions", "Tasks", "Specialists", "Tools", "Knowledge"):
        await expect(sidebar.get_by_role("link", name=label, exact=True)).to_be_visible()
    await expect(sidebar.get_by_role("link", name="System", exact=True)).to_have_count(0)
    await expect(sidebar.get_by_role("link", name="Start a session", exact=True)).to_have_count(0)

    # System explains itself without reading diagnostics. Evals stay readable, with
    # their execution settings explained rather than shown.
    await page.goto(f"{base_url}/cayu/system", wait_until="networkidle")
    await expect(page.get_by_test_id("dashboard-role-unavailable")).to_contain_text(
        ROLE_FORBIDDEN_REASON
    )
    require_equal(system_reads, [], "a member must not read System diagnostics")
    await _exercise_remaining_reads(page, base_url)
    await page.goto(f"{base_url}/cayu/evals/settings", wait_until="networkidle")
    await expect(
        page.get_by_text("Your role can read eval suites and comparisons", exact=False)
    ).to_be_visible()
    await expect(
        page.get_by_role("heading", name="What trials run with", exact=True)
    ).to_have_count(0)
    await page.goto(f"{base_url}/cayu/evals", wait_until="networkidle")
    await expect(page.get_by_role("heading", name="Evals", exact=True)).to_be_visible()

    # A session offers decisions but no composer, Run again or changed arguments.
    await page.goto(f"{base_url}/cayu/sessions/{MEMBER_APPROVAL_SESSION}", wait_until="networkidle")
    card = page.locator("[data-inbox-item]").first
    await expect(card.get_by_role("button", name="Approve", exact=True)).to_be_visible()
    await expect(page.locator("#session-composer")).to_have_count(0)
    await card.get_by_text("Decision options", exact=True).click()
    await expect(card.get_by_label("Approve with changed arguments", exact=True)).to_have_count(0)
    await page.get_by_role("button", name="More actions", exact=True).click()
    await expect(page.get_by_role("button", name="Run again", exact=True)).to_have_count(0)
    await page.keyboard.press("Escape")

    # Tasks and Agent pages read correctly and offer no writes.
    await page.goto(f"{base_url}/cayu/tasks", wait_until="networkidle")
    await expect(page.get_by_role("link", name="Member reads this task")).to_be_visible()
    await expect(page.get_by_role("button", name="Create a task", exact=True)).to_have_count(0)
    await page.goto(f"{base_url}/cayu/tasks/{MEMBER_TASK}", wait_until="networkidle")
    await expect(
        page.get_by_text("Your role can read tasks; task changes require full access.", exact=True)
    ).to_be_visible()
    await expect(page.get_by_role("button", name="Cancel task", exact=True)).to_have_count(0)
    await page.get_by_role("button", name="More actions", exact=True).click()
    await expect(page.get_by_role("button", name="Pause", exact=True)).to_have_count(0)
    await page.keyboard.press("Escape")
    await page.goto(f"{base_url}/cayu/specialists", wait_until="networkidle")
    await page.get_by_role("button", name=f"Inspect {AGENT_NAME}", exact=True).click()
    await expect(page.get_by_text("System prompt", exact=True)).to_be_visible()
    await expect(page.get_by_role("link", name="Start a session")).to_have_count(0)
    await page.goto(f"{base_url}/cayu/tools", wait_until="networkidle")
    await page.get_by_role("button", name=f"Inspect {TOOL_NAME}", exact=True).click()
    await expect(page.get_by_text("Input schema", exact=True)).to_be_visible()
    await page.goto(f"{base_url}/cayu/knowledge", wait_until="networkidle")
    member_kept = page.get_by_role("article", name=f"Knowledge {MEMBER_KEPT}", exact=True)
    await expect(member_kept.get_by_text(ROLE_FORBIDDEN_REASON, exact=True)).to_be_visible()
    await expect(member_kept.get_by_role("button", name="Remove", exact=True)).to_have_count(0)
    proposal = page.locator("[data-inbox-item]").filter(has_text="Members keep proposed knowledge.")
    async with page.expect_response(_settle_response) as kept:
        await proposal.get_by_role("button", name="Keep", exact=True).click()
    require_equal((await kept.value).status, 200, "a member may keep a knowledge proposal")

    # A page that still believes it has full access (its role changed after it loaded)
    # shows the writes; the server refuses each one and the page says why.
    async with failures.forbidden_allowed() as forced:
        forged = await _believe_full_access(page, signer)
        try:
            await page.goto(f"{base_url}/cayu/", wait_until="networkidle")
            await page.get_by_role("button", name="Start a session", exact=True).click()
            await page.locator("#start-specialist").select_option(AGENT_NAME)
            await page.locator("#start-prompt").fill("Member forced start.")
            await page.get_by_role("button", name="Start", exact=True).click()
            await expect(page.get_by_text("Request rejected", exact=True)).to_be_visible()
            await expect(page.get_by_text(ROLE_FORBIDDEN_REASON, exact=False).first).to_be_visible()
            await expect(page.get_by_text("Something went wrong", exact=False)).to_have_count(0)

            await page.goto(f"{base_url}/cayu/tasks/{MEMBER_TASK}", wait_until="networkidle")
            await page.get_by_role("button", name="More actions", exact=True).click()
            await page.get_by_role("button", name="Pause", exact=True).click()
            await page.get_by_role("button", name="Pause task", exact=True).click()
            await expect(page.get_by_role("alert")).to_have_text(ROLE_FORBIDDEN_REASON)
            require_equal(
                await _task_status(tasks, MEMBER_TASK),
                TaskStatus.PENDING,
                "a refused member hold must leave the task unchanged",
            )

            await page.goto(
                f"{base_url}/cayu/sessions/{MEMBER_APPROVAL_SESSION}", wait_until="networkidle"
            )
            card = page.locator("[data-inbox-item]").first
            await card.get_by_text("Decision options", exact=True).click()
            await card.get_by_label("Approve with changed arguments", exact=True).check()
            await card.get_by_label("Changed arguments", exact=True).fill(
                json.dumps(FORCED_ARGUMENTS)
            )
            await card.get_by_role("button", name="Approve", exact=True).click()
            await expect(card.get_by_role("alert")).to_have_text(ROLE_FORBIDDEN_REASON)
        finally:
            await page.unroute("**/api/app", forged)
        # The page learns the real role again on its next identity read and keeps the
        # draft, offering the original arguments instead.
        async with page.expect_response(lambda response: urlsplit(response.url).path == "/api/app"):
            await page.evaluate("""() => {
              Object.defineProperty(document, 'visibilityState', {configurable: true, value: 'hidden'})
              window.dispatchEvent(new Event('visibilitychange'))
              Object.defineProperty(document, 'visibilityState', {configurable: true, value: 'visible'})
              window.dispatchEvent(new Event('visibilitychange'))
            }""")
        await expect(
            card.get_by_text("Your current role cannot change arguments", exact=False)
        ).to_be_visible()
        await expect(card.get_by_role("button", name="Approve", exact=True)).to_be_disabled()

        # The server refuses the same writes however they arrive.
        refusals = await page.evaluate(
            """async ({ taskId, sessionId, args }) => {
              const post = async (path, body) => {
                const response = await fetch(path, {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify(body),
                })
                return { status: response.status, detail: (await response.json()).detail }
              }
              const inbox = await (await fetch(`/api/inbox?session_id=${sessionId}`)).json()
              return {
                hold: await post(`/api/tasks/${taskId}/pause`, {}),
                run: await post("/api/run", {
                  agent_name: "dashboard-roles-agent",
                  messages: [{ role: "user", content: [{ type: "text", text: "Forced run." }] }],
                }),
                settle: await post(`/api/inbox/${inbox.items[0].id}/settle`, {
                  decision: "approve",
                  arguments: args,
                }),
              }
            }""",
            {
                "taskId": MEMBER_TASK,
                "sessionId": MEMBER_APPROVAL_SESSION,
                "args": FORCED_ARGUMENTS,
            },
        )
    for name, refusal in refusals.items():
        require_equal(refusal["status"], 403, f"a member's forced {name} must be refused")
        require_equal(
            refusal["detail"].get("code"), "role_forbidden", f"a member's forced {name} reason"
        )
    # Believing it has full access, the page also makes full-access reads (a running
    # session's message queue, say); the server refuses those too.
    for method, path in forced:
        require(
            method == "GET"
            or path == "/api/run"
            or path == f"/api/tasks/{MEMBER_TASK}/pause"
            or re.fullmatch(r"/api/inbox/[^/]+/settle", path) is not None,
            f"only the forced writes may be refused, not {method} {path}",
        )
    require(
        {path for _method, path in forced} >= {"/api/run", f"/api/tasks/{MEMBER_TASK}/pause"},
        f"the forced writes must reach the server: {forced}",
    )

    # Inbox decisions stay open to members: approve, deny and answer.
    for session_id, decision in (
        (MEMBER_APPROVAL_SESSION, "Approve"),
        (MEMBER_DENIAL_SESSION, "Deny"),
    ):
        await page.goto(f"{base_url}/cayu/inbox?session_id={session_id}", wait_until="networkidle")
        card = page.locator("[data-inbox-item]").first
        async with page.expect_response(_settle_response) as settled:
            await card.get_by_role("button", name=decision, exact=True).click()
        require_equal((await settled.value).status, 200, f"a member may {decision.lower()}")
    await page.goto(
        f"{base_url}/cayu/inbox?session_id={MEMBER_QUESTION_SESSION}", wait_until="networkidle"
    )
    card = page.locator("[data-inbox-item]").first
    await expect(
        card.get_by_text("Which report should the agent send?", exact=True)
    ).to_be_visible()
    async with page.expect_response(_settle_response) as answered:
        await card.get_by_role("button", name="weekly", exact=True).click()
    require_equal((await answered.value).status, 200, "a member may answer a question")
    for session_id in (MEMBER_APPROVAL_SESSION, MEMBER_QUESTION_SESSION):
        await _wait_for_status(sessions, session_id, SessionStatus.COMPLETED)
    require(
        (MEMBER_APPROVAL_SESSION, ORIGINAL_ARGUMENTS) in tool.runs
        and all(args != FORCED_ARGUMENTS for _session, args in tool.runs),
        f"a member's approval must run the original arguments only: {tool.runs}",
    )
    require(
        all(session != MEMBER_DENIAL_SESSION for session, _args in tool.runs),
        "a denied call must not run",
    )
    return forced


async def _believe_full_access(
    page: Page, signer: IdentitySigner
) -> Callable[[Route], Awaitable[None]]:
    """Answer `/api/app` with full access while every other request keeps the real role."""

    async def full_access(route: Route) -> None:
        headers = {**await route.request.all_headers(), IDENTITY_HEADER: signer.token("member")}
        response = await route.fetch(headers=headers)
        body = await response.json()
        body.update(role="admin", full_access=True)
        await route.fulfill(response=response, json=body)

    await page.route("**/api/app", full_access)
    return full_access


def _settle_response(response) -> bool:
    return (
        response.request.method == "POST"
        and re.fullmatch(r"/api/inbox/[^/]+/settle", urlsplit(response.url).path) is not None
    )


async def _wait_for_status(
    sessions: InMemorySessionStore, session_id: str, status: SessionStatus
) -> None:
    for _ in range(200):
        session = await sessions.load(session_id)
        if session is not None and session.status == status:
            return
        await asyncio.sleep(0.05)
    session = await sessions.load(session_id)
    raise AssertionError(f"{session_id} stayed {session and session.status}, expected {status}")


async def _signed_context(
    browser: Browser, base_url: str, signer: IdentitySigner, role: str
) -> BrowserContext:
    """A browser whose every request to the server carries a fresh identity for `role`."""

    context = await browser.new_context(viewport={"width": 1440, "height": 1000}, locale="en-US")

    async def sign(route: Route) -> None:
        headers = {**await route.request.all_headers(), IDENTITY_HEADER: signer.token(role)}
        await route.fallback(headers=headers)

    await context.route(f"{base_url}/**", sign)
    return context


async def _capture(*pages: Page) -> None:
    directory = Path(
        os.environ.get(
            "CAYU_DASHBOARD_DIAGNOSTICS_DIR",
            str(Path(tempfile.gettempdir()) / "cayu-dashboard-behavior"),
        )
    )
    directory.mkdir(parents=True, exist_ok=True)
    for index, page in enumerate(pages):
        try:
            await page.screenshot(
                path=str(directory / f"dashboard-roles-{index}.png"), full_page=True
            )
        except Exception as exc:
            print(f"dashboard roles screenshot {index} failed: {exc}")
    print(f"CAYU_DASHBOARD_ROLES_DIAGNOSTICS={directory}")


@asynccontextmanager
async def _serve(app) -> AsyncIterator[str]:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.setblocking(False)
    port = listener.getsockname()[1]
    # sse_starlette remembers process-wide that an earlier server in this process shut
    # down and then ends every new event stream at once; this server is a new start.
    AppStatus.should_exit = False
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="on"))
    task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        while not server.started:
            if task.done():
                await task
                raise RuntimeError("dashboard roles server stopped before startup")
            await asyncio.sleep(0.01)
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, timeout=10)
        except TimeoutError:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


if __name__ == "__main__":
    print(json.dumps(asyncio.run(exercise_role_access()), indent=2, sort_keys=True))
