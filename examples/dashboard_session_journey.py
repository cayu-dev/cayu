"""One session-workspace journey through the Developer dashboard.

A person starts a session, watches it run, leaves and comes back, reviews an approval
it pauses on, resumes it, sees it complete and reads it back after a reload. Along the
way the check holds the workspace to its promises: an unsent draft and the reading
position survive navigation and reloads, the document itself never scrolls, the
composer keeps its place on desktop, and the sidebar choice survives routes and
reloads. The model is deterministic and waits for the check at each step, so every
state is observed rather than raced.

Saved browser playback isn't covered here. Recordings reach the dashboard only through
`BrowserRecordingServer`, whose `BrowserRecordingStore` needs FFmpeg and Pillow
(`cayu[recording]`) to start and accepts media only as frames from the capture guest
over `wss://`, which it encodes with FFmpeg; and the session's Browser tab appears only
when the native browser tool has left its browser-usage receipt, which applications
can't write. `examples/browser_recording/` covers recording and dashboard playback
with Docker and FFmpeg.

Run it alone with `CAYU_DASHBOARD_PART=journey python examples/dashboard_behavior_live.py`.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import socket
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

import uvicorn
from playwright.async_api import FloatRect, Locator, Page, async_playwright, expect
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
from cayu.messages import MessageRole, TextPart, ToolResultPart
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.server import BasicAuth, ServerConfig, create_server
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionStatus
from cayu.tools import AlwaysRequireApprovalToolPolicy

AGENT_NAME = "dashboard-journey-agent"
MODEL_NAME = "dashboard-journey-model"
TOOL_NAME = "dashboard_journey_review"
OUTLINE_TOOL_NAME = "dashboard_journey_outline"
OPERATOR = ("dashboard-journey-operator", "dashboard-journey-password")
PROMPT = "Prepare the dashboard journey report."
REPORT_PARAGRAPHS = 40
FINAL_TEXT = "The dashboard journey report is complete."
DRAFT = "Journey draft: keep the summary to three lines."
FOLLOW_UP_TEXT = "Follow-up received for the journey."
DRAFT_KEY_PREFIX = "cayu-dashboard-draft:"
EARLIER_SESSIONS = 40
EARLIER_PROMPT = "Earlier dashboard journey session"

# The conversation's scrolling element: the session page's one overflowing column.
_CONVERSATION = """() => [...document.querySelectorAll('main div')].find(
  (node) => getComputedStyle(node).overflowY === 'auto' && node.querySelector('#session-composer') === null
    && node.scrollHeight > node.clientHeight
)"""


# The report section whose text is at the top of the conversation's visible area.
_TOP_SECTION = f"""() => {{
  const node = ({_CONVERSATION})()
  if (!node) return null
  const box = node.getBoundingClientRect()
  for (let y = box.top + 4; y < box.top + 120; y += 4) {{
    const hit = document.caretRangeFromPoint(box.left + box.width / 2, y)
    const text = hit?.startContainer?.nodeType === Node.TEXT_NODE ? hit.startContainer.data : ''
    const start = text.lastIndexOf('Report section ', hit?.startOffset ?? 0)
    if (start >= 0) return Number(/^Report section (\\d+)/.exec(text.slice(start))[1])
  }}
  return null
}}"""


class JourneyProvider(ModelProvider):
    """Writes a long report, asks to review it, then finishes.

    Writing the report and finishing each wait for `release`, so the check sees the
    session running. The report goes with an outline call that needs no approval, so
    it's in the conversation before the review pauses the session.
    """

    name = "dashboard-journey-provider"

    def __init__(self) -> None:
        self.started = 0
        self._started = asyncio.Condition()
        self._releases: asyncio.Queue[None] = asyncio.Queue()

    async def wait_for_calls(self, count: int) -> None:
        async with self._started:
            await asyncio.wait_for(self._started.wait_for(lambda: self.started >= count), 15)

    def release(self) -> None:
        self._releases.put_nowait(None)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        last = request.messages[-1]
        if last.role == MessageRole.USER and any(
            isinstance(part, TextPart) and part.text.startswith(EARLIER_PROMPT)
            for part in last.content
        ):
            yield ModelStreamEvent.text_delta("Earlier session finished.")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})
            return
        if last.role == MessageRole.USER and any(
            isinstance(part, TextPart) and DRAFT in part.text for part in last.content
        ):
            yield ModelStreamEvent.text_delta(FOLLOW_UP_TEXT)
            yield ModelStreamEvent.completed({"finish_reason": "stop"})
            return
        results = {part.tool_name for part in last.content if isinstance(part, ToolResultPart)}
        if OUTLINE_TOOL_NAME in results:
            yield ModelStreamEvent.tool_call(
                id="dashboard-journey-review-call",
                name=TOOL_NAME,
                arguments={"section": "summary"},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        async with self._started:
            self.started += 1
            self._started.notify_all()
        await asyncio.wait_for(self._releases.get(), timeout=60)
        if TOOL_NAME in results:
            yield ModelStreamEvent.text_delta(FINAL_TEXT)
            yield ModelStreamEvent.completed({"finish_reason": "stop"})
            return
        report = "\n\n".join(
            f"Report section {index}: the journey checks this paragraph stays readable."
            for index in range(1, REPORT_PARAGRAPHS + 1)
        )
        yield ModelStreamEvent.text_delta(report)
        yield ModelStreamEvent.tool_call(
            id="dashboard-journey-outline-call",
            name=OUTLINE_TOOL_NAME,
            arguments={"sections": REPORT_PARAGRAPHS},
        )
        yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})


class JourneyReviewTool(Tool):
    spec = ToolSpec(
        name=TOOL_NAME,
        description="Publish the reviewed report section; it needs a person's approval.",
        input_schema={
            "type": "object",
            "properties": {"section": {"type": "string"}},
            "required": ["section"],
        },
        effect=ToolEffect.NONE,
        max_terminal_payload_bytes=1024,
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(content=f"Published the {args['section']} section.")


class JourneyOutlineTool(Tool):
    spec = ToolSpec(
        name=OUTLINE_TOOL_NAME,
        description="Outline the report; it runs without review.",
        input_schema={
            "type": "object",
            "properties": {"sections": {"type": "integer"}},
            "required": ["sections"],
        },
        effect=ToolEffect.NONE,
        max_terminal_payload_bytes=1024,
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(content=f"Outlined {args['sections']} sections.")


async def _box(locator: Locator) -> FloatRect:
    box = await locator.bounding_box()
    require(box is not None, "the element has no layout box")
    assert box is not None
    return box


async def exercise_session_journey() -> dict[str, object]:
    sessions = InMemorySessionStore()
    app = CayuApp(session_store=sessions, enable_logging=False)
    provider = JourneyProvider()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name=AGENT_NAME, model=MODEL_NAME),
        tools=[JourneyOutlineTool(), JourneyReviewTool()],
        tool_policy=AlwaysRequireApprovalToolPolicy(tools=[TOOL_NAME]),
    )
    # Enough finished sessions that the Sessions list scrolls.
    for index in range(EARLIER_SESSIONS):
        async for _ in app.run(
            RunRequest(
                agent_name=AGENT_NAME,
                messages=[Message.text("user", f"{EARLIER_PROMPT} {index}.")],
            )
        ):
            pass
    server_app = create_server(
        app,
        config=ServerConfig.protected(BasicAuth(username=OPERATOR[0], password=OPERATOR[1])),
    )
    failures: list[str] = []
    async with _serve(server_app) as base_url, async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            context = await browser.new_context(
                viewport={"width": 1440, "height": 1000},
                http_credentials={"username": OPERATOR[0], "password": OPERATOR[1]},
                locale="en-US",
            )
            # Count scrolls of the document itself; the dashboard scrolls inside its panes.
            await context.add_init_script(
                """window.__documentScrolls = 0
                document.addEventListener('scroll', (event) => {
                  if (event.target === document) window.__documentScrolls += 1
                }, true)"""
            )
            page = await context.new_page()
            expected_failures: set[str] = set()
            _watch(page, failures, expected_failures)
            try:
                evidence = await _journey(page, base_url, provider, sessions, expected_failures)
                require(not failures, f"the session journey recorded browser failures: {failures}")
            except BaseException:
                await _capture(page)
                raise
        finally:
            await browser.close()
    return evidence


async def _journey(
    page: Page,
    base_url: str,
    provider: JourneyProvider,
    sessions: InMemorySessionStore,
    expected_failures: set[str],
) -> dict[str, object]:
    # Start: the form follows the new session live while the model works.
    await page.goto(f"{base_url}/cayu/run", wait_until="networkidle")
    await page.locator("#start-specialist").select_option(AGENT_NAME)
    await page.locator("#start-prompt").fill(PROMPT)
    await page.get_by_role("button", name="Start", exact=True).click()
    await provider.wait_for_calls(1)
    await expect(page.get_by_text("Live stream", exact=True)).to_be_visible()
    await _require_document_still(page, "while the session starts")

    # Human review: the session pauses on an approval and the form opens it.
    provider.release()
    await expect(page).to_have_url(re.compile(r"/cayu/sessions/[^/?]+$"), timeout=15000)
    session_id = urlsplit(page.url).path.rsplit("/", 1)[-1]
    session_url = page.url
    card = page.locator("[data-inbox-item]")
    await expect(card.get_by_role("button", name="Approve", exact=True)).to_be_visible()
    await expect(page.get_by_text("Needs your approval", exact=True).first).to_be_visible()
    composer = page.locator("#session-composer")
    await composer.fill(DRAFT)
    await composer.blur()
    resting = await _composer_geometry(page)
    reading = await _read_from_middle(page)
    await _require_document_still(page, "while reviewing")

    # Navigation: leave for the list and come back by its link; then leave for the Inbox
    # and come back with Back. Each return keeps the draft and the reading position.
    await page.get_by_role("link", name="Sessions", exact=True).first.click()
    await expect(page).to_have_url(re.compile(r"/cayu/sessions$"))
    await page.locator(f'a[href$="/sessions/{session_id}"]').first.click()
    await expect(page).to_have_url(session_url)
    await _require_returned(page, reading, resting, "after the Sessions list")
    await page.get_by_role("navigation", name="Main").get_by_role("link", name="Inbox").click()
    await expect(page).to_have_url(re.compile(r"/cayu/inbox$"))
    await expect(page.locator("[data-inbox-item]").first).to_be_visible()
    await page.go_back()
    await expect(page).to_have_url(session_url)
    await _require_returned(page, reading, resting, "after Back from the Inbox")
    await page.reload(wait_until="networkidle")
    await _require_returned(page, reading, resting, "after a reload")

    # Resume: approving runs the session again; the draft waits in the composer.
    async with page.expect_response(
        lambda response: (
            response.request.method == "POST"
            and re.fullmatch(r"/api/inbox/[^/]+/settle", urlsplit(response.url).path) is not None
        )
    ) as settled:
        await card.get_by_role("button", name="Approve", exact=True).click()
    require_equal((await settled.value).status, 200, "approving the journey's review")
    await provider.wait_for_calls(2)
    await expect(page.get_by_text("Working…", exact=True)).to_be_visible()
    await expect(page.get_by_role("button", name="Stop", exact=True)).to_be_visible()
    await expect(composer).to_have_value(DRAFT)
    require_equal(await _composer_geometry(page), resting, "the composer moved while running")
    await _require_document_still(page, "while running")

    # Navigating away while it runs, the list shows it running; coming back keeps the draft.
    await page.get_by_role("link", name="Sessions", exact=True).first.click()
    row = page.get_by_role("row").filter(has=page.locator(f'a[href$="/sessions/{session_id}"]'))
    await expect(row.get_by_text("Running", exact=True)).to_be_visible()
    await row.locator(f'a[href$="/sessions/{session_id}"]').click()
    await expect(page).to_have_url(session_url)
    await expect(composer).to_have_value(DRAFT)

    # Completion: a reader at the end of the conversation follows the final answer in.
    box = await _box(composer)
    await page.mouse.move(box["x"] + box["width"] / 2, box["y"] - 200)
    await page.mouse.wheel(0, 20000)
    await page.wait_for_function(
        f"""() => {{
          const node = ({_CONVERSATION})()
          return node && node.scrollHeight - node.scrollTop - node.clientHeight <= 2
        }}"""
    )
    provider.release()
    await _wait_for_status(sessions, session_id, SessionStatus.COMPLETED)
    await expect(page.get_by_text(FINAL_TEXT, exact=True)).to_be_visible(timeout=15000)
    await expect(page.get_by_text("Completed", exact=True).first).to_be_visible()
    await expect(composer).to_have_value(DRAFT)
    require_equal(await _composer_geometry(page), resting, "the composer moved after completion")

    # A definitive rejected reply must remain a draft across a reload.
    async def reject_reply(route):
        await route.fulfill(status=403, json={"detail": "Reply refused for this check."})

    expected_failures.add("/api/resume")
    await page.route("**/api/resume", reject_reply)
    try:
        async with page.expect_response(
            lambda response: urlsplit(response.url).path == "/api/resume"
        ):
            await composer.press("Enter")
        await page.wait_for_function(
            "([key, text]) => sessionStorage.getItem(key) === text",
            arg=[DRAFT_KEY_PREFIX + session_id, DRAFT],
        )
        await page.reload(wait_until="networkidle")
        await expect(composer).to_have_value(DRAFT)
    finally:
        await page.unroute("**/api/resume", reject_reply)
        expected_failures.discard("/api/resume")

    # A saved pixel position from another transcript window must not be restored.
    await _read_from_middle(page)
    await page.evaluate(
        """(key) => {
          const saved = JSON.parse(sessionStorage.getItem(key))
          if (!saved || saved.transcriptOffset === undefined) throw Error('missing window identity')
          saved.transcriptOffset += 20
          sessionStorage.setItem(key, JSON.stringify(saved))
        }""",
        "cayu-dashboard-reading:session:" + session_id,
    )
    await page.reload(wait_until="networkidle")
    await page.wait_for_function(
        f"""() => {{
          const node = ({_CONVERSATION})()
          return node && node.scrollHeight - node.scrollTop - node.clientHeight <= 32
        }}"""
    )

    # The draft is sent as a follow-up, and sending forgets it.
    await composer.press("Enter")
    await expect(page.get_by_text(FOLLOW_UP_TEXT, exact=True)).to_be_visible(timeout=15000)
    await expect(composer).to_have_value("")
    require(
        await page.evaluate(f"sessionStorage.getItem({json.dumps(DRAFT_KEY_PREFIX + session_id)})")
        is None,
        "a sent draft must be forgotten",
    )
    await _require_document_still(page, "after the follow-up")

    # Saved: after a reload the whole conversation reads back from the server. The
    # session used no browser, so its details offer no Browser tab or playback.
    await page.reload(wait_until="networkidle")
    await expect(page.get_by_text(FOLLOW_UP_TEXT, exact=True)).to_be_visible()
    for text in (PROMPT, "Report section 1:", FINAL_TEXT, DRAFT, FOLLOW_UP_TEXT):
        await expect(page.get_by_text(text, exact=False).first).to_be_attached()
    await expect(composer).to_have_value("")
    await page.get_by_role("button", name="Session details", exact=True).click()
    details = page.get_by_role("complementary", name="Session details")
    await expect(details.get_by_role("tab", name="Overview", exact=True)).to_be_visible()
    await expect(details.get_by_role("tab", name="Browser", exact=True)).to_have_count(0)
    await page.get_by_role("button", name="Session details", exact=True).click()

    await _exercise_list_position(page)
    await _exercise_sidebar_preference(page, session_url)
    await _exercise_narrow_screen(page)
    await _require_document_still(page, "at the end of the journey")
    return {
        "browser": "chromium",
        "session_id": session_id,
        "model_calls": provider.started,
        "interactions": [
            "start_running",
            "human_review_pause",
            "draft_kept_across_list_inbox_back_and_reload",
            "reading_position_kept_across_list_inbox_back_and_reload",
            "resume_running",
            "navigation_while_running",
            "completion",
            "rejected_reply_draft_survives_reload",
            "shifted_transcript_window_discards_saved_offset",
            "follow_up_clears_draft",
            "saved_conversation_after_reload",
            "list_position_kept_on_back",
            "sidebar_preference_across_routes_and_reloads",
            "stable_desktop_composer",
            "no_document_scrolling",
            "narrow_screen_navigation",
        ],
        "saved_playback": "not_covered",
    }


async def _read_from_middle(page: Page) -> int:
    """Scroll the conversation up from its end, as a reader would; return the section read."""

    box = await _box(page.locator("#session-composer"))
    await page.mouse.move(box["x"] + box["width"] / 2, box["y"] - 200)
    await page.mouse.wheel(0, -900)
    await page.wait_for_function(
        f"""() => {{
          const node = ({_CONVERSATION})()
          return node && node.scrollHeight - node.scrollTop - node.clientHeight > 400
        }}"""
    )
    # Wheel scrolling can glide; read the position once it has settled.
    top = None
    for _ in range(50):
        await page.wait_for_timeout(100)
        current = await page.evaluate(f"({_CONVERSATION})().scrollTop")
        if current == top:
            break
        top = current
    section = await page.evaluate(_TOP_SECTION)
    require(
        isinstance(section, int) and 1 < section < REPORT_PARAGRAPHS,
        f"the reader must be in the middle of the report, not at section {section}",
    )
    return section


async def _require_returned(page: Page, section: int, resting: dict, label: str) -> None:
    """The draft is back and the same report section is at the top of the conversation."""

    await expect(page.locator("#session-composer")).to_have_value(DRAFT)
    try:
        await page.wait_for_function(f"(section) => ({_TOP_SECTION})() === section", arg=section)
    except Exception:
        current = await page.evaluate(_TOP_SECTION)
        raise AssertionError(f"reading section {section} was lost {label}; at {current}") from None
    require_equal(await _composer_geometry(page), resting, f"the composer moved {label}")
    await _require_document_still(page, label)


async def _composer_geometry(page: Page) -> dict:
    """Where the composer sits in the window, rounded to whole pixels."""

    return await page.evaluate("""() => {
      const box = document.querySelector('#session-composer').parentElement.getBoundingClientRect()
      return {
        left: Math.round(box.left),
        width: Math.round(box.width),
        height: Math.round(box.height),
        bottomGap: Math.round(innerHeight - box.bottom),
      }
    }""")


async def _require_document_still(page: Page, label: str) -> None:
    state = await page.evaluate("""() => ({
      scrolls: window.__documentScrolls,
      scrollY: window.scrollY,
      overflow: document.documentElement.scrollHeight - innerHeight,
    })""")
    require_equal(state["scrolls"], 0, f"the document scrolled {label}")
    require_equal(state["scrollY"], 0, f"the document is scrolled {label}")
    require(state["overflow"] <= 0, f"the document overflows the window {label}: {state}")


async def _exercise_list_position(page: Page) -> None:
    """Back returns the Sessions list to where it was read; a new visit starts at the top."""

    sidebar = page.get_by_role("navigation", name="Main")
    await sidebar.get_by_role("link", name="Sessions", exact=True).click()
    await expect(page.get_by_text(f"{EARLIER_PROMPT} 0.", exact=False).first).to_be_attached()
    list_body = """() => [...document.querySelectorAll('main div')].find(
      (node) => getComputedStyle(node).overflowY === 'auto' && node.scrollHeight > node.clientHeight
    )"""
    box = await _box(page.locator("main"))
    await page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    await page.mouse.wheel(0, 700)
    top = None
    for _ in range(50):
        await page.wait_for_timeout(100)
        current = await page.evaluate(f"({list_body})()?.scrollTop ?? 0")
        if current == top:
            break
        top = current
    require(top is not None and top > 0, "the Sessions list must scroll")
    await sidebar.get_by_role("link", name="Inbox", exact=True).click()
    await expect(page).to_have_url(re.compile(r"/cayu/inbox$"))
    await page.go_back()
    await expect(page).to_have_url(re.compile(r"/cayu/sessions$"))
    await page.wait_for_function(
        f"(top) => Math.abs((({list_body})()?.scrollTop ?? 0) - top) <= 2", arg=top
    )
    await _require_document_still(page, "after Back to the Sessions list")
    await sidebar.get_by_role("link", name="Inbox", exact=True).click()
    await sidebar.get_by_role("link", name="Sessions", exact=True).click()
    await expect(page.get_by_text(f"{EARLIER_PROMPT} 0.", exact=False).first).to_be_attached()
    await page.wait_for_timeout(300)
    require_equal(
        await page.evaluate(f"({list_body})()?.scrollTop ?? 0"),
        0,
        "following a link to the Sessions list starts at the top",
    )


async def _exercise_sidebar_preference(page: Page, session_url: str) -> None:
    """Hiding the sidebar on desktop lasts across routes and reloads, and so does showing it."""

    await page.goto(session_url, wait_until="networkidle")
    sidebar = page.get_by_role("complementary", name="Sidebar")
    await expect(sidebar).to_be_visible()
    await sidebar.get_by_role("button", name="Hide sidebar", exact=True).click()
    await expect(sidebar).to_be_hidden()
    show = page.get_by_role("button", name="Show sidebar", exact=True)
    await expect(show).to_be_focused()
    await page.get_by_role("link", name="Sessions", exact=True).first.click()
    await expect(page).to_have_url(re.compile(r"/cayu/sessions$"))
    await expect(sidebar).to_be_hidden()
    await page.reload(wait_until="networkidle")
    await expect(sidebar).to_be_hidden()
    await page.goto(session_url, wait_until="networkidle")
    await expect(sidebar).to_be_hidden()
    await show.click()
    await expect(sidebar).to_be_visible()
    await page.reload(wait_until="networkidle")
    await expect(sidebar).to_be_visible()
    await expect(show).to_have_count(0)


async def _exercise_narrow_screen(page: Page) -> None:
    await page.set_viewport_size({"width": 390, "height": 844})
    try:
        await expect(page.locator("#session-composer")).to_be_in_viewport()
        require(
            await page.evaluate("document.documentElement.scrollWidth <= innerWidth"),
            "the session must fit a phone-width window",
        )
        sidebar = page.get_by_role("dialog", name="Sidebar")
        await page.get_by_role("button", name="Open sidebar", exact=True).click()
        await expect(sidebar).to_be_visible()
        await expect(
            sidebar.get_by_role("button", name="Close sidebar", exact=True)
        ).to_be_focused()
        await page.keyboard.press("Escape")
        await expect(sidebar).to_have_count(0)
    finally:
        await page.set_viewport_size({"width": 1440, "height": 1000})


async def _wait_for_status(
    sessions: InMemorySessionStore, session_id: str, status: SessionStatus
) -> None:
    for _ in range(300):
        session = await sessions.load(session_id)
        if session is not None and session.status == status:
            return
        await asyncio.sleep(0.05)
    session = await sessions.load(session_id)
    raise AssertionError(f"{session_id} stayed {session and session.status}, expected {status}")


def _watch(page: Page, failures: list[str], expected_failures: set[str]) -> None:
    page.on("pageerror", lambda error: failures.append(f"pageerror: {error}"))

    def console(message) -> None:
        if message.type != "error":
            return
        if (
            "status of 403" in message.text
            and urlsplit(message.location["url"]).path in expected_failures
        ):
            return
        # The app has no session-message policy, so the queue probe on a running
        # session is refused; that response is expected.
        if "status of 403" in message.text and re.fullmatch(
            r"/api/sessions/[^/]+/messages", urlsplit(message.location["url"]).path
        ):
            return
        failures.append(f"console: {message.text}")

    def response(response) -> None:
        path = urlsplit(response.url).path
        if not path.startswith("/api/") or response.status < 400:
            return
        if response.status == 403 and path in expected_failures:
            return
        if (
            response.status == 403
            and response.request.method == "GET"
            and re.fullmatch(r"/api/sessions/[^/]+/messages", path)
        ):
            return
        failures.append(f"{response.request.method} {path}: HTTP {response.status}")

    page.on("console", console)
    page.on("response", response)


async def _capture(page: Page) -> None:
    directory = Path(
        os.environ.get(
            "CAYU_DASHBOARD_DIAGNOSTICS_DIR",
            str(Path(tempfile.gettempdir()) / "cayu-dashboard-behavior"),
        )
    )
    directory.mkdir(parents=True, exist_ok=True)
    try:
        await page.screenshot(path=str(directory / "dashboard-journey.png"), full_page=True)
    except Exception as exc:
        print(f"dashboard journey screenshot failed: {exc}")
    print(f"CAYU_DASHBOARD_JOURNEY_DIAGNOSTICS={directory}")


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
                raise RuntimeError("dashboard journey server stopped before startup")
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
    print(json.dumps(asyncio.run(exercise_session_journey()), indent=2, sort_keys=True))
