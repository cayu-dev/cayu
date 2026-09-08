"""Rendered production operator component against a bounded API fixture.

Requires an explicitly started local dashboard Vite server, not a deployed backend.
"""

import asyncio
import os
from urllib.parse import urlsplit

import pytest
from tests.core.test_browser_control import operator_purpose

pytestmark = pytest.mark.skipif(
    not os.environ.get("CAYU_BROWSER_DASHBOARD_URL"), reason="Opt-in dashboard rendering."
)

_OPERATOR_HTML = """<div id="root"></div><script type="module">
import RefreshRuntime from '/@react-refresh';
RefreshRuntime.injectIntoGlobalHook(window);
window.$RefreshReg$ = () => {}; window.$RefreshSig$ = () => type => type;
window.__vite_plugin_react_preamble_installed__ = true;
window.__CAYU_DASHBOARD_CONFIG__ = {apiBaseUrl:'https://operator.test/api'};
const React = (await import('/node_modules/.vite/deps/react.js')).default;
const {createRoot} = (await import('/node_modules/.vite/deps/react-dom_client.js')).default;
const {BrowserOperator} = await import('/src/components/dashboard/browser-operator.tsx');
createRoot(document.getElementById('root')).render(
  React.createElement(BrowserOperator, {sessionId:'session'}));
</script>"""


@pytest.mark.parametrize("policy", ["unavailable", "on_close"])
@pytest.mark.parametrize("observation_end", ["failure", "expiry", "selection"])
def test_rendered_checkpoint_consent_is_bound_to_takeover(policy, observation_end):
    from playwright.async_api import async_playwright, expect

    async def scenario():
        root = os.environ["CAYU_BROWSER_DASHBOARD_URL"].rstrip("/")
        browser_record = {
            "identity": {
                "profile_checkpoint_policy": policy,
                "operator_purpose": operator_purpose().model_dump(mode="json"),
                "session_id": "session",
                "session_instance_id": "instance",
                "run_epoch": 1,
                "interaction_id": "interaction",
                "execution_profile_fingerprint": "a" * 64,
                "environment_name": "browser",
                "allocation_fingerprint": "b" * 64,
                "browser_session_id": "browser-session",
                "worker_instance_id": "worker",
            },
            "revision": 2,
            "control_epoch": 1,
            "state": "agent_controlled",
            "sensitive_entry": False,
            "sensitive_entry_pending": False,
            "fresh_observation_required": False,
            "owned_request": None,
        }
        takeovers = []
        requests = []
        page_reads = 0
        delayed_read = asyncio.Event()
        release_read = asyncio.Event()
        delivered_read = asyncio.Event()

        async def api(route):
            nonlocal page_reads
            path = route.request.url.rsplit("/", 1)[-1]
            requests.append((path, route.request.post_data))
            if path == "operator-session":
                result = {"operator_session_token": "rendered-continuity"}
            elif path == "session":
                result = {"browsers": [browser_record]}
            elif path == "pages":
                page_reads += 1
                read_number = page_reads
                if read_number == 2 and observation_end == "failure":
                    await route.fulfill(status=503, json={"detail": "private-failure-canary"})
                    return
                if read_number == 2 and observation_end == "selection":
                    delayed_read.set()
                    await release_read.wait()
                page_descriptor = {"page_id": "page", "revision": "revision", "control_epoch": 1}
                result = {
                    "pages": [page_descriptor],
                    "locations": [
                        {
                            "page": page_descriptor,
                            "origin": "https://late-origin.test"
                            if observation_end == "selection" and read_number == 2
                            else "https://current-origin.test",
                        }
                    ],
                    "active_page_id": "page",
                }
            elif path == "takeover":
                submitted = route.request.post_data_json
                takeovers.append(submitted)
                browser_record.update(
                    state="operator_controlled",
                    revision=4,
                    control_epoch=2,
                    owned_request={
                        "request_id": submitted["request_id"],
                        "expires_at_ms": submitted["expires_at_ms"],
                        "maximum_until_ms": submitted["maximum_until_ms"],
                        "lease_until_ms": submitted["expires_at_ms"],
                        "pending_lease_until_ms": None,
                        "settled_input_sequence": 0,
                        "pending_input_sequence": None,
                        "checkpoint_consent": submitted["checkpoint_consent"],
                    },
                )
                result = {}
            elif path == "sensitive-entry":
                browser_record.update(sensitive_entry_pending=True, revision=5)
                result = {}
            elif path == "handback":
                browser_record.update(state="handback_pending", revision=7)
                result = {}
            else:
                raise AssertionError("Unexpected operator API call")
            await route.fulfill(
                json=result,
                headers={
                    "Access-Control-Allow-Origin": root,
                    "Access-Control-Allow-Credentials": "true",
                    "Access-Control-Allow-Headers": "content-type,x-cayu-browser-operator",
                },
            )
            if path == "pages" and page_reads >= 2 and release_read.is_set():
                delivered_read.set()

        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                executable_path=os.environ.get("CAYU_BROWSER_CONTROL_CHROMIUM"), headless=True
            )
            try:
                page = await browser.new_page()
                await page.add_init_script(
                    "window.__operatorClock = 0; performance.now = () => window.__operatorClock;"
                )
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.on(
                    "console",
                    lambda message: (
                        errors.append(message.text) if message.type == "error" else None
                    ),
                )
                await page.route("https://operator.test/api/browser-control/**", api)
                await page.route(
                    f"{root}/operator-test",
                    lambda route: route.fulfill(
                        content_type="text/html",
                        body=_OPERATOR_HTML,
                    ),
                )
                await page.goto(f"{root}/operator-test")
                try:
                    await page.get_by_role("button", name="Discover browsers").wait_for(
                        timeout=5000
                    )
                except Exception:
                    raise AssertionError(errors) from None
                await page.get_by_role("button", name="Discover browsers").click()
                await page.get_by_role("button", name="Browser 1 · agent_controlled").click()
                origin = page.get_by_text(
                    "Page 1 observed origin: https://current-origin.test.", exact=True
                )
                await expect(origin).to_be_visible()
                if observation_end == "selection":
                    await asyncio.wait_for(delayed_read.wait(), 5)
                    await page.get_by_role("button", name="Browser 1 · agent_controlled").click()
                    release_read.set()
                    await asyncio.wait_for(delivered_read.wait(), 5)
                    await page.evaluate("() => new Promise(resolve => setTimeout(resolve, 100))")
                    await expect(origin).to_be_visible()
                    await expect(page.get_by_text("late-origin.test", exact=False)).to_have_count(0)
                else:
                    if observation_end == "expiry":
                        await page.evaluate("window.__operatorClock = 300000")
                    await expect(
                        page.get_by_text(
                            "Live locations are unavailable. Refresh browser discovery.", exact=True
                        )
                    ).to_be_visible(timeout=5000)
                    await expect(origin).to_have_count(0)
                    assert "private-failure-canary" not in await page.locator("body").inner_text()
                    await page.get_by_role("button", name="Browser 1 · agent_controlled").click()
                    await expect(origin).to_be_visible()
                consent = page.get_by_label("Profile checkpoint consent for the next takeover")
                await expect(consent).to_have_value("undecided")
                allow = consent.locator('option[value="allow"]')
                if policy == "unavailable":
                    await expect(allow).to_be_disabled()
                    selected = "deny"
                else:
                    await expect(allow).to_be_enabled()
                    selected = "allow"
                await consent.select_option(selected)
                await page.get_by_role("button", name="Request exclusive takeover").click()
                await expect(
                    page.get_by_role("button", name="Refresh control state")
                ).to_be_enabled()
                assert len(takeovers) == 1
                assert takeovers[0]["checkpoint_consent"] == selected
                assert takeovers[0]["identity"] == browser_record["identity"]
                private_value = page.get_by_label("Private value", exact=True)
                await expect(private_value).to_have_count(0)
                await page.get_by_role("button", name="Prepare sensitive entry").click()
                await expect(
                    page.get_by_role("button", name="Prepare sensitive entry")
                ).to_be_disabled()
                await expect(private_value).to_have_count(0)
                await expect(
                    page.get_by_role("button", name="Return control to agent")
                ).to_be_disabled()
                # Supply positive native/purge settlement, then require fresh
                # page discovery before any rendered input control is usable.
                browser_record.update(
                    sensitive_entry_pending=False, sensitive_entry=True, revision=6
                )
                await page.get_by_role("button", name="Refresh control state").click()
                await expect(private_value).to_be_visible()
                send = page.get_by_role("button", name="Send private text once")
                await expect(send).to_be_disabled()
                await page.get_by_role("button", name="Browser 1 · operator_controlled").click()
                await expect(send).to_be_enabled()
                await private_value.fill("unsent-private-render-canary")
                await page.get_by_role("button", name="Return control to agent").click()
                await expect(private_value).to_have_count(0)
                await expect(
                    page.get_by_role("button", name="Return control to agent")
                ).to_be_disabled()
                assert "unsent-private-render-canary" not in repr(requests)
                await page.get_by_role("button", name="Close private view").click()
                await expect(page.get_by_role("status")).to_contain_text(
                    "does not hand control back"
                )
            finally:
                release_read.set()
                await browser.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("authenticated", "allowed"), [(False, False), (True, False), (True, True)]
)
def test_rendered_discovery_uses_production_authentication(tmp_path, authenticated, allowed):
    import httpx
    from playwright.async_api import async_playwright, expect
    from pydantic import SecretBytes
    from tests.core.test_browser_control_authorization import Policy
    from tests.core.test_browser_control_publisher import publication_fixture

    from cayu import CayuApp
    from cayu.runtime._browser_control_publisher import BrowserControlPublisher
    from cayu.runtime.browser_control_config import BrowserControlConfig
    from cayu.server import BrowserControlServerConfig, ServerConfig, create_server
    from cayu.server.auth import BasicAuth

    async def scenario():
        root = os.environ["CAYU_BROWSER_DASHBOARD_URL"].rstrip("/")
        async with publication_fixture("memory", tmp_path) as (store, bootstrap):
            await BrowserControlPublisher(store).publish(bootstrap)
            app = CayuApp(
                enable_logging=False,
                session_store=store._store,
                browser_control=BrowserControlConfig(
                    purpose=operator_purpose(),
                    policy=Policy(allowed),
                    guest_endpoint="wss://operator.test/api/browser-control/guest",
                ),
            )
            server = create_server(
                app,
                config=ServerConfig.protected(
                    BasicAuth(username="operator", password="password"),
                    browser_control=BrowserControlServerConfig(
                        operator_origin="https://operator.test", signing_key=SecretBytes(b"k" * 32)
                    ),
                ),
            )
            responses = []
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="https://operator.test"
            ) as client:

                async def forward(route):
                    path = urlsplit(route.request.url).path
                    if path.startswith("/api/"):
                        response = await client.request(
                            route.request.method,
                            path,
                            headers=await route.request.all_headers(),
                            content=route.request.post_data_buffer,
                        )
                        responses.append((path, response.status_code))
                        await route.fulfill(
                            status=response.status_code,
                            headers={
                                key: value
                                for key, value in response.headers.items()
                                if key not in {"content-encoding", "content-length"}
                            },
                            body=response.content,
                        )
                    elif path == "/operator-test":
                        await route.fulfill(content_type="text/html", body=_OPERATOR_HTML)
                    else:
                        response = await route.fetch(url=f"{root}{path}")
                        await route.fulfill(response=response)

                async with async_playwright() as playwright:
                    browser = await playwright.chromium.launch(
                        executable_path=os.environ.get("CAYU_BROWSER_CONTROL_CHROMIUM"),
                        headless=True,
                    )
                    try:
                        page = await browser.new_page(
                            extra_http_headers=(
                                {"Authorization": "Basic b3BlcmF0b3I6cGFzc3dvcmQ="}
                                if authenticated
                                else {}
                            )
                        )
                        await page.route("https://operator.test/**", forward)
                        await page.goto("https://operator.test/operator-test")
                        await page.get_by_role("button", name="Discover browsers").click()
                        await expect(page.get_by_role("status")).not_to_contain_text(
                            "requires explicit application authorization"
                        )
                        expected_status = (
                            "Select an authorized browser and page"
                            if authenticated and allowed
                            else "did not settle locally"
                        )
                        assert expected_status in await page.get_by_role("status").inner_text(), (
                            responses
                        )
                        assert responses[0] == (
                            "/api/browser-control/operator-session",
                            200 if authenticated else 401,
                        )
                        assert len(responses) == (2 if authenticated else 1)
                        if authenticated:
                            assert responses[1] == (
                                "/api/browser-control/sessions/session",
                                200 if allowed else 403,
                            )
                        await expect(
                            page.get_by_role("button", name="Request exclusive takeover")
                        ).to_have_count(0)
                    finally:
                        await browser.close()

    asyncio.run(scenario())
