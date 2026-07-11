"""Verified browser contract for the packaged Cayu dashboard."""

from __future__ import annotations

import asyncio
import json
import os
import re
import socket
import tempfile
from pathlib import Path

import uvicorn
from playwright.async_api import (  # ty: ignore[unresolved-import]
    BrowserContext,
    Page,
    async_playwright,
    expect,
)

from _live_checks import require
from cayu import CayuApp, Event, EventType, Message, RunRequest
from cayu.runtime import InMemorySessionStore, SessionIdentity, SessionStatus
from cayu.server import create_server

SESSION_ID = "dashboard-contract-session"
AGENT_NAME = "dashboard-contract-agent"
PROVIDER_NAME = "contract-provider"
MODEL_NAME = "contract-model"
PAYLOAD_MARKER = "dashboard-contract-usage"
EVIDENCE_PREFIX = "CAYU_NIGHTLY_EVIDENCE="


async def main() -> None:
    app = await _seed_app()
    listener = _loopback_listener()
    port = listener.getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"
    server = uvicorn.Server(
        uvicorn.Config(
            create_server(app, dev=True),
            log_level="warning",
            lifespan="off",
        )
    )
    server_task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        await _wait_for_server(server, server_task)
        evidence = await _run_browser_contract(base_url)
        print(f"{EVIDENCE_PREFIX}{json.dumps(evidence, sort_keys=True)}")
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(server_task, timeout=10)
        except TimeoutError:
            server_task.cancel()
            await asyncio.gather(server_task, return_exceptions=True)


async def _seed_app() -> CayuApp:
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)
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
                id="dashboard-session-completed",
                type=EventType.SESSION_COMPLETED,
                session_id=SESSION_ID,
                agent_name=AGENT_NAME,
            ),
        ],
    )
    await store.update_status(SESSION_ID, SessionStatus.COMPLETED)
    return app


async def _run_browser_contract(base_url: str) -> dict[str, object]:
    browser_failures: dict[str, list[str]] = {
        "console_errors": [],
        "page_errors": [],
        "request_failures": [],
        "api_errors": [],
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
        await context.tracing.start(screenshots=True, snapshots=True)
        page = await context.new_page()
        _record_browser_failures(page, browser_failures)
        try:
            await _exercise_dashboard(page, base_url)
            _require_no_browser_failures(browser_failures)
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
        "interactions": ["sessions_list", "session_detail", "event_detail"],
        "console_errors": 0,
        "page_errors": 0,
        "request_failures": 0,
        "api_errors": 0,
    }


async def _exercise_dashboard(page: Page, base_url: str) -> None:
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


def _record_browser_failures(page: Page, failures: dict[str, list[str]]) -> None:
    page.on(
        "console",
        lambda message: (
            failures["console_errors"].append(message.text) if message.type == "error" else None
        ),
    )
    page.on("pageerror", lambda error: failures["page_errors"].append(str(error)))
    page.on(
        "requestfailed",
        lambda request: failures["request_failures"].append(
            f"{request.method} {request.url}: {request.failure or 'unknown failure'}"
        ),
    )
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
