"""Browser-smoke an ejected dashboard build against the installed Cayu server."""

from __future__ import annotations

import argparse
import socket
import threading
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import uvicorn
from playwright.sync_api import sync_playwright

from cayu import CayuApp
from cayu.server import DashboardConfig, ServerConfig, create_server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dashboard_build", type=Path)
    args = parser.parse_args(argv)
    dashboard_build = args.dashboard_build.resolve(strict=True)

    app = create_server(
        CayuApp(),
        config=ServerConfig.local_development(
            dashboard=DashboardConfig(path="/operator", directory=dashboard_build)
        ),
    )
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="off"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    try:
        _wait_for_server(base_url)
        observed_reads: set[str] = set()
        browser_errors: list[str] = []
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.on(
                "response",
                lambda response: (
                    observed_reads.add(response.url)
                    if response.status == 200
                    and response.url.endswith(("/api/contract", "/api/system/diagnostics"))
                    else None
                ),
            )
            page.on("pageerror", lambda error: browser_errors.append(str(error)))
            page.goto(f"{base_url}/operator/system", wait_until="networkidle")
            page.get_by_text("Bounded Cayu configuration snapshot", exact=True).wait_for()
            if page.locator('[data-testid="dashboard-contract-gate"]').count() != 0:
                raise RuntimeError(
                    "dashboard server-contract compatibility gate rejected the server"
                )
            browser.close()

        expected_reads = {f"{base_url}/api/contract", f"{base_url}/api/system/diagnostics"}
        if observed_reads != expected_reads:
            raise RuntimeError(
                f"dashboard did not complete the expected bounded reads: {sorted(observed_reads)}"
            )
        if browser_errors:
            raise RuntimeError(f"dashboard browser errors: {browser_errors}")
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        listener.close()
    print("validated ejected dashboard deep link, contract gate, and bounded system read")
    return 0


def _wait_for_server(base_url: str) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            with urlopen(f"{base_url}/api/health", timeout=1) as response:
                if response.status == 200:
                    return
        except (OSError, URLError):
            time.sleep(0.05)
    raise RuntimeError("Cayu server did not become ready for the dashboard smoke test")


if __name__ == "__main__":
    raise SystemExit(main())
