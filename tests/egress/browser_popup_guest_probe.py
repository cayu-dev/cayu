"""Run the real guest popup pipeline inside the pinned Chromium test container."""

from __future__ import annotations

import asyncio
import json
import runpy
import sys
from urllib.parse import urlsplit

from playwright.async_api import async_playwright


async def main():
    worker = runpy.run_path("/opt/cayu-browser/worker.py", run_name="popup_probe_worker")
    configuration = json.load(sys.stdin)
    case = configuration["case"]
    limits = worker["_InteractiveLimits"](**configuration["limits"])
    policy = worker["_InteractivePopupPolicy"]("same_origin", ("click",), (), ())
    scripts = {
        "blank": "const child=window.open('about:blank'); child.location='https://example.test/Upper?q=Case#Fragment'",
        "direct": "window.open('https://example.test/Upper?q=Case#Fragment')",
        "reordered": "window.open('https://example.test/First#One'); window.open('https://example.test/Second#Two')",
    }
    root = (
        '<form method="post" target="_blank" action="https://example.test/submit">'
        '<input name="token" value="body"><button>Open</button></form>'
        if case == "post"
        else f'<button onclick="{scripts[case]}">Open</button>'
    )
    dispatched = []
    daemon = worker["_InteractiveDaemon"]("bs_probe")

    def request(operation, **updates):
        return worker["_InteractiveRequest"](
            **{
                "operation": operation,
                "operation_id": operation,
                "session_id": "bs_probe",
                "page_id": "bp_root",
                "expected_revision": None,
                "expected_control_epoch": None,
                "ref": None,
                "url": None,
                "value": None,
                "key": None,
                "wait_ms": None,
                "full_page": False,
                "limits": limits,
                "multi_page": True,
                "popup_policy": policy,
                **updates,
            }
        )

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True, args=["--no-sandbox"])
        context = await browser.new_context(service_workers="block")
        daemon.browser = browser
        daemon.context = context
        daemon.browser_version = browser.version
        try:
            opening = request("navigate", url="https://example.test/root")
            await daemon._ensure_configuration(opening)

            class Route:
                def __init__(self, original):
                    self.original = original

                async def abort(self, reason):
                    await self.original.abort(reason)

                async def continue_(self):
                    incoming = self.original.request
                    dispatched.append((incoming.method, incoming.url))
                    body = (
                        root
                        if urlsplit(incoming.url).path == "/root"
                        else "<title>Child</title>Child"
                    )
                    await self.original.fulfill(status=200, content_type="text/html", body=body)

            async def route(incoming):
                await daemon._route_interactive_request(Route(incoming), incoming.request)

            await context.route("**/*", route)
            if case == "reordered":
                note = daemon._note_popup_candidate
                pending = []

                def reversed_candidates(page):
                    if page in pending:
                        return
                    pending.append(page)
                    if len(pending) == 2:
                        for candidate in reversed(pending):
                            note(candidate)

                daemon._note_popup_candidate = reversed_candidates
            opened = await daemon.execute(opening)
            assert opened["kind"] == "success", opened
            observation = opened["observation"]
            ref = next(item["ref"] for item in observation["refs"] if item["name"] == "Open")
            result = await daemon.execute(
                request(
                    "click",
                    ref=ref,
                    expected_revision=observation["revision"],
                    expected_control_epoch=observation["control_epoch"],
                )
            )
            if case == "post":
                assert result["kind"] == "error" and result["error"] == "policy_denied", result
                assert dispatched == [("GET", "https://example.test/root")], dispatched
            else:
                assert result["kind"] == "success", result
                children = [page for page in result["page_set"]["pages"] if page["opener_page_id"]]
                urls = {page["url"] for page in children}
                expected = (
                    {"https://example.test/First#One", "https://example.test/Second#Two"}
                    if case == "reordered"
                    else {"https://example.test/Upper?q=Case#Fragment"}
                )
                assert urls == expected, (urls, result)
                assert all(page["lifecycle"] == "background" for page in children)
                assert len(dispatched) == len(expected) + 1, dispatched
                replay = await daemon.execute(
                    request(
                        "click",
                        ref=ref,
                        expected_revision=observation["revision"],
                        expected_control_epoch=observation["control_epoch"],
                    )
                )
                assert replay == result
                assert len(dispatched) == len(expected) + 1
            print(json.dumps({"case": case, "passed": True, "requests": dispatched}))
        finally:
            assert await daemon.close(), "Browser owner failed to settle"


if __name__ == "__main__":
    asyncio.run(main())
