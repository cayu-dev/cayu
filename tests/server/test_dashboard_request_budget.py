"""Opt-in request budget for every Developer dashboard page.

Each page is opened in the packaged dashboard, left to settle, and watched for a minute.
The test fails when a page starts more API requests than `dashboard/tests/request-budgets.json`
allows, when it re-reads an endpoint that sends an ETag without revalidating (an
unchanged 200 where a 304 was possible), or when leaving the page leaves its requests
running. It runs twice: idle, and with one session making progress every second.

    CAYU_REQUEST_BUDGET_LIVE=1 uv run pytest tests/server/test_dashboard_request_budget.py

`CAYU_REQUEST_BUDGET_PAGES=home,session` limits the pages. Pass `CAYU_SESSION_SEARCH_DIRECTORY`
to serve a dashboard build other than the packaged one.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("playwright.async_api")

from tests.core.test_cloud_reporting import PRICE, USAGE, _app, _drain
from tests.server.test_dashboard_session_search import search_browser

from cayu import AgentSpec, Message
from cayu.messages import MessageRole
from cayu.providers import ModelRequest, ModelStreamEvent
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sessions import RunRequest
from cayu.tools.base import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec

pytestmark = pytest.mark.skipif(
    os.environ.get("CAYU_REQUEST_BUDGET_LIVE") != "1", reason="Opt-in packaged UI request budget."
)

BUDGETS = json.loads(
    (Path(__file__).parents[2] / "dashboard" / "tests" / "request-budgets.json").read_text()
)
WINDOW_SECONDS = BUDGETS["windowSeconds"]
SETTLE_SECONDS = 5
QUIET_SECONDS = 2
MAX_SETTLE_SECONDS = 20
# Reads the shell keeps for every page; they aren't the page's to cancel.
SHELL_PATHS = ("/api/contract", "/api/app", "/api/inbox/counts")
FINISHED = "finished-session"
RUNNING = "running-session"


class _Tick(Tool):
    spec = ToolSpec(
        name="tick",
        description="Wait a second.",
        input_schema={"type": "object", "properties": {}},
        effect=ToolEffect.IDEMPOTENT,
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="tests:request-budget:tick", behavior_version="1", implementation_version="1"
        ),
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        await asyncio.sleep(1)
        return ToolResult(content="Ticked.")


def _tick_forever(request: ModelRequest) -> list[ModelStreamEvent]:
    """Call `tick` again after every result: one model step and one tool call per second."""
    rounds = sum(message.role is MessageRole.TOOL for message in request.messages)
    if "tick" not in {tool["name"] for tool in request.tools}:
        return [
            ModelStreamEvent.text_delta("Done."),
            ModelStreamEvent.completed({"finish_reason": "stop", "usage": USAGE}),
        ]
    return [
        ModelStreamEvent.text_delta(f"Round {rounds}."),
        ModelStreamEvent.tool_call(id=f"tick-{rounds}", name="tick", arguments={}),
        ModelStreamEvent.completed({"finish_reason": "tool_calls", "usage": USAGE}),
    ]


async def _runtime():
    from tests.core.test_cloud_reporting import _Provider

    runtime = _app()
    runtime.register_provider(_Provider(name="ticker", response_factory=_tick_forever))
    runtime.register_agent(
        AgentSpec(name="ticker", model=PRICE.model, provider_name="ticker"), tools=[_Tick()]
    )
    await _drain(
        runtime.run(
            RunRequest(
                session_id=FINISHED,
                agent_name="writer",
                title="Finished session",
                messages=[Message.text("user", "Draft release notes")],
            )
        )
    )
    return runtime


@dataclass
class _Request:
    url: str
    method: str
    started: float
    ended: float | None = None
    failed: bool = False
    status: int | None = None
    etag: str | None = None
    digest: str | None = None


@dataclass
class _Recorder:
    requests: list[_Request] = field(default_factory=list)
    by_handle: dict[int, _Request] = field(default_factory=dict)

    def attach(self, page) -> None:
        page.on("request", self._started)
        page.on("requestfinished", self._finished)
        page.on("requestfailed", self._failed)

    def _started(self, request) -> None:
        if "/api/" not in request.url:
            return
        row = _Request(url=request.url, method=request.method, started=time.monotonic())
        self.requests.append(row)
        self.by_handle[id(request)] = row

    async def _finished(self, request) -> None:
        row = self.by_handle.get(id(request))
        if row is None:
            return
        response = await request.response()
        row.ended = time.monotonic()
        if response is None:
            return
        row.status = response.status
        row.etag = await response.header_value("etag")
        if response.status == 200 and "event-stream" not in (
            await response.header_value("content-type") or ""
        ):
            try:
                row.digest = hashlib.sha256(await response.body()).hexdigest()
            except Exception:
                row.digest = None

    def _failed(self, request) -> None:
        row = self.by_handle.get(id(request))
        if row is not None:
            row.ended = time.monotonic()
            row.failed = True


def _path(url: str) -> str:
    return "/" + url.split("/", 3)[3].split("?", 1)[0]


def _unrevalidated(rows: list[_Request]) -> list[str]:
    """GETs answered with the same 200 body as before although the endpoint sent an ETag."""
    seen: dict[str, _Request] = {}
    repeats = []
    for row in rows:
        if row.method != "GET" or row.status != 200 or row.digest is None:
            continue
        before = seen.get(row.url)
        if before is not None and before.etag and before.digest == row.digest:
            repeats.append(row.url)
        seen[row.url] = row
    return repeats


async def _navigate_in_app(page, path: str) -> None:
    """Client-side navigation, so the left page's queries are cancelled, not torn down."""
    await page.evaluate(
        "path => { history.pushState(null, '', path); dispatchEvent(new PopStateEvent('popstate')) }",
        path,
    )


async def _settle(page, recorder: _Recorder) -> None:
    """Wait for the page's loading reads: at least SETTLE_SECONDS, then until no API read
    has started for QUIET_SECONDS (at most MAX_SETTLE_SECONDS in all)."""
    begun = time.monotonic()
    await page.wait_for_timeout(SETTLE_SECONDS * 1000)
    while time.monotonic() - begun < MAX_SETTLE_SECONDS:
        last = max((row.started for row in recorder.requests), default=begun)
        if time.monotonic() - last >= QUIET_SECONDS:
            return
        await page.wait_for_timeout(250)


async def _measure(page, base: str, name: str, spec: dict, running: bool) -> list[str]:
    budget = spec["running" if running else "idle"]
    if budget is None:  # not measured in this scenario
        return []
    path = spec["path"].replace("{session}", RUNNING if running else FINISHED)
    recorder = _Recorder()
    recorder.attach(page)
    await page.goto(f"{base}/cayu{path}")
    await _settle(page, recorder)
    start = time.monotonic()
    await page.wait_for_timeout(WINDOW_SECONDS * 1000)
    window = [row for row in recorder.requests if row.started >= start]
    problems = []
    if len(window) > budget:
        counts: dict[str, int] = {}
        for row in window:
            counts[_path(row.url)] = counts.get(_path(row.url), 0) + 1
        problems.append(f"{name}: {len(window)} requests in {WINDOW_SECONDS}s > {budget}: {counts}")
    repeats = _unrevalidated(recorder.requests)
    if repeats:
        problems.append(f"{name}: re-read unchanged ETag responses without revalidating: {repeats}")
    left = time.monotonic()
    pending = [row for row in recorder.requests if row.ended is None]
    await _navigate_in_app(page, "/cayu/system")
    await page.wait_for_timeout(3000)
    leaked = [
        row.url
        for row in pending
        if not row.failed
        and _path(row.url) not in SHELL_PATHS
        and (row.ended is None or row.ended - left > 1.0)
    ]
    if leaked:
        problems.append(f"{name}: requests kept running after leaving the page: {leaked}")
    print(f"{name} ({'running' if running else 'idle'}): {len(window)}/{budget} requests")
    for problem in problems:
        print(f"  {problem}")
    return problems


def _pages() -> list[tuple[str, dict]]:
    wanted = {name for name in os.environ.get("CAYU_REQUEST_BUDGET_PAGES", "").split(",") if name}
    return [
        (name, spec)
        for name, spec in BUDGETS["pages"].items()
        if (not wanted or name in wanted) and "{" not in spec["path"].replace("{session}", "")
    ]


async def _stop(task: asyncio.Task | None) -> None:
    if task is not None and not task.done():
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


@pytest.mark.parametrize("running", [False, True], ids=["idle", "running"])
def test_every_page_stays_within_its_request_budget(running):
    async def run():
        runtime = await _runtime()
        ticking = None
        if running:
            ticking = asyncio.create_task(
                _drain(
                    runtime.run(
                        RunRequest(
                            session_id=RUNNING,
                            agent_name="ticker",
                            title="Running session",
                            messages=[Message.text("user", "Tick")],
                        )
                    )
                )
            )
        try:
            problems = []
            async with search_browser(runtime=runtime) as (page, base, *_):
                page.set_default_timeout(15_000)
                for name, spec in _pages():
                    problems += await _measure(page, base, name, spec, running)
                # Stop the session before the server shuts down, so nothing holds it open.
                await _stop(ticking)
            assert not problems, "\n".join(problems)
        finally:
            await _stop(ticking)

    asyncio.run(run())
