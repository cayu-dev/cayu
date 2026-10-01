from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib
import io
import json
import os
import re
import wave
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI, WebSocket
from fastapi.responses import Response
from tests.core.test_browser_control_transport import control_tls as _control_tls
from tests.server._browser_control_tls_server import browser_control_tls_server

from cayu import default_price_book
from cayu.server.static import DashboardStaticFiles, dashboard_content_security_policy

control_tls = _control_tls


@pytest.mark.parametrize(
    ("api_base_url", "sources"),
    [
        ("/api", ["'self'"]),
        ("https://api.example/api", ["'self'", "https://api.example", "wss://api.example"]),
        (
            "https://api.example:8443/api",
            ["'self'", "https://api.example:8443", "wss://api.example:8443"],
        ),
        ("http://localhost:8000/api", ["'self'", "http://localhost:8000"]),
    ],
)
def test_dashboard_csp_scopes_connections_to_the_configured_api(api_base_url, sources):
    policy = dashboard_content_security_policy(config_script="", api_base_url=api_base_url)
    connect = next(part for part in policy.split("; ") if part.startswith("connect-src "))
    assert connect.split()[1:] == sources
    media = next(part for part in policy.split("; ") if part.startswith("media-src "))
    assert media.split()[1:] == ["'self'", "blob:"] + [
        source for source in sources if source.startswith(("http://", "https://"))
    ]


def test_no_pricebook_browser_fixture_retains_a_matching_script_hash(tmp_path, monkeypatch):
    pytest.importorskip("playwright.async_api")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples"))
    example = importlib.import_module("dashboard_behavior_live")
    (tmp_path / "index.html").write_text("<html><head></head><body></body></html>")
    dashboard = DashboardStaticFiles(
        directory=str(tmp_path),
        base_path="/cayu",
        api_base_url="https://api.example/api",
        dashboard_config={"priceBook": default_price_book()},
    )
    original = dashboard._index_response({"type": "http", "method": "GET"})

    class Response:
        headers = dict(original.headers)

        async def text(self):
            return original.body.decode()

    class Route:
        rewritten = None

        async def fetch(self):
            return Response()

        async def fulfill(self, **kwargs):
            self.rewritten = kwargs

    route = Route()
    asyncio.run(example._serve_dashboard_without_pricebook(route))
    assert route.rewritten is not None
    script = re.search(r"<script>(.*?)</script>", route.rewritten["body"], re.S).group(1)
    config = json.loads(script.removeprefix("window.__CAYU_DASHBOARD_CONFIG__=").removesuffix(";"))
    assert "priceBook" not in config
    assert config["apiBaseUrl"] == "https://api.example/api"
    digest = base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
    policy = route.rewritten["headers"]["content-security-policy"]
    assert f"'sha256-{digest}'" in policy
    assert policy != original.headers["content-security-policy"]
    assert "'unsafe-inline'" not in policy.split("script-src ", 1)[1].split(";", 1)[0]
    assert "wss://api.example" in policy


@pytest.mark.skipif(
    os.environ.get("CAYU_BROWSER_CONTROL_LIVE") != "1",
    reason="Opt-in isolated Chromium acceptance.",
)
@pytest.mark.parametrize("resource", ["control-socket", "recording"])
def test_dashboard_csp_allows_cross_origin_browser_resources(tmp_path, control_tls, resource):
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization
    from playwright.async_api import async_playwright

    async def scenario():
        app = FastAPI()

        @app.websocket("/api/browser-control/viewer")
        async def viewer(socket: WebSocket):
            await socket.accept()
            await socket.send_text("viewer-connected")
            await socket.close()

        data = io.BytesIO()
        with wave.open(data, "wb") as recording:
            recording.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
            recording.writeframes(b"\0\0" * 8000)

        @app.get("/api/browser-recordings/recording/media")
        async def recording_media():
            return Response(data.getvalue(), media_type="audio/wav")

        certificate = x509.load_pem_x509_certificate((tmp_path / "certificate.pem").read_bytes())
        pin = base64.b64encode(
            hashlib.sha256(
                certificate.public_key().public_bytes(
                    serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
                )
            ).digest()
        ).decode()
        (tmp_path / "index.html").write_text("<html><head></head><body></body></html>")
        async with browser_control_tls_server(app, tmp_path) as port:
            api = f"https://127.0.0.1:{port}/api"
            dashboard = DashboardStaticFiles(directory=str(tmp_path), api_base_url=api)
            frontend = FastAPI()
            frontend.mount("/", dashboard)
            async with (
                browser_control_tls_server(frontend, tmp_path) as frontend_port,
                async_playwright() as playwright,
            ):
                browser = await playwright.chromium.launch(
                    executable_path=os.environ.get("CAYU_BROWSER_CONTROL_CHROMIUM"),
                    headless=True,
                    args=["--ignore-certificate-errors-spki-list=" + pin],
                )
                try:
                    page = await browser.new_page()
                    await page.goto(f"https://127.0.0.1:{frontend_port}/")
                    if resource == "control-socket":
                        script = """url => new Promise((resolve, reject) => {
                            const socket = new WebSocket(url);
                            socket.onmessage = event => resolve(event.data);
                            socket.onerror = () => reject(new Error("Viewer socket blocked"));
                        })"""
                        url = api.replace("https:", "wss:") + "/browser-control/viewer"
                        expected = "viewer-connected"
                    else:
                        script = """url => new Promise((resolve, reject) => {
                            const video = document.createElement("video");
                            video.onloadedmetadata = () => resolve("recording-loaded");
                            video.onerror = () => reject(new Error("Recording blocked"));
                            video.src = url;
                            video.preload = "auto";
                            document.body.append(video);
                            video.load();
                        })"""
                        url = api + "/browser-recordings/recording/media"
                        expected = "recording-loaded"
                    result = await asyncio.wait_for(page.evaluate(script, url), timeout=10)
                    assert result == expected
                finally:
                    await browser.close()

    asyncio.run(scenario())
