"""Rendered operator journey against ASGI routes and a live native command owner."""

import asyncio
import json
import os
from urllib.parse import urlsplit

from tests.server.test_browser_operator_dashboard_live import _OPERATOR_HTML


async def rendered_native_operator_journey(
    browser,
    client,
    api,
    owner,
    daemon,
    *,
    mounted_api=False,
    clock_ms=1000,
    session_id="session",
    private_text="tls-native-canary",
    network_socket_origin=None,
    network_tls=None,
    dashboard_path=None,
    expected_page_origin=None,
    changed_page_origin=None,
    keyboard_only=False,
    failures=None,
):
    from playwright.async_api import WebSocketRoute, expect

    root = os.environ.get("CAYU_BROWSER_DASHBOARD_URL", "").rstrip("/")
    assert dashboard_path is not None or root
    token = None
    responses = []
    sockets = []
    protocol = []
    purged = asyncio.Event()
    release_purge = asyncio.Event()
    hold_purge = False
    initial_operations = daemon.total_operations
    phase = "discovery"

    def api_path(path):
        return path if mounted_api else path.removeprefix("/api")

    async def forward_socket(socket: WebSocketRoute):
        incoming = asyncio.Queue()
        await incoming.put({"type": "websocket.connect"})

        def receive(payload):
            protocol.append(("receive", type(payload).__name__, len(payload)))
            message = {
                "type": "websocket.receive",
                "text" if isinstance(payload, str) else "bytes": payload,
            }
            if isinstance(payload, str) and payload.startswith("purged:"):
                purged.set()

                async def deliver_ack():
                    if hold_purge:
                        await release_purge.wait()
                    await incoming.put(message)

                sockets.append(asyncio.create_task(deliver_ack()))
            else:
                incoming.put_nowait(message)

        socket.on_message(receive)

        if network_socket_origin is not None:
            from websockets.asyncio.client import connect
            from websockets.typing import Origin, Subprotocol

            from cayu.tools._browser_control_transport import _private_transport_logger

            assert network_tls is not None

            async def relay_network():
                async with connect(
                    network_socket_origin + api_path(urlsplit(socket.url).path),
                    ssl=network_tls,
                    origin=Origin("https://operator.test"),
                    subprotocols=[Subprotocol(value) for value in socket.protocols],
                    proxy=None,
                    compression=None,
                    max_size=2 * 1024 * 1024,
                    max_queue=1,
                    open_timeout=5,
                    close_timeout=2,
                    logger=_private_transport_logger(),
                ) as connection:

                    async def deliver():
                        while True:
                            message = await incoming.get()
                            if message["type"] == "websocket.receive":
                                await connection.send(message.get("text", message.get("bytes")))

                    delivery = asyncio.create_task(deliver())
                    try:
                        async for payload in connection:
                            protocol.append(("network-send", type(payload).__name__, len(payload)))
                            socket.send(payload)
                            await asyncio.sleep(0)
                    finally:
                        delivery.cancel()
                        await asyncio.gather(delivery, return_exceptions=True)
                    await socket.close(code=1000, reason="")

            sockets.append(asyncio.create_task(relay_network()))
            return

        async def send(message):
            protocol.append(("send", message["type"]))
            if message["type"] == "websocket.send":
                socket.send(message.get("text", message.get("bytes")))
                # Playwright's synchronous send schedules its transport task;
                # let it enqueue before forwarding the following ASGI close.
                await asyncio.sleep(0)
            elif message["type"] == "websocket.close":
                await socket.close(code=message.get("code", 1000), reason="")

        scope = {
            "type": "websocket",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "scheme": "wss",
            "path": api_path(urlsplit(socket.url).path),
            "root_path": "",
            "query_string": b"",
            "subprotocols": socket.protocols,
            "headers": [(b"origin", b"https://operator.test")],
            "server": ("operator.test", 443),
            "client": ("127.0.0.1", 1234),
        }
        sockets.append(asyncio.create_task(api(scope, incoming.get, send)))

    async def forward(route):
        nonlocal token
        path = urlsplit(route.request.url).path
        if path.startswith("/api/"):
            assert private_text.encode() not in (route.request.post_data_buffer or b"")
            result = await client.request(
                route.request.method,
                api_path(path),
                headers=await route.request.all_headers(),
                content=route.request.post_data_buffer,
            )
            assert private_text.encode() not in result.content
            if path.endswith("/operator-session") and result.status_code == 200:
                token = result.json()["operator_session_token"]
            responses.append((path, result.status_code))
            await route.fulfill(
                status=result.status_code,
                headers={
                    key: value
                    for key, value in result.headers.items()
                    if key not in {"content-encoding", "content-length"}
                },
                body=result.content,
            )
        elif dashboard_path is not None:
            result = await client.get(path, headers=await route.request.all_headers())
            assert result.status_code == 200
            await route.fulfill(
                status=result.status_code,
                headers={
                    key: value
                    for key, value in result.headers.items()
                    if key not in {"content-encoding", "content-length"}
                },
                body=result.content,
            )
        elif path == "/operator-test":
            await route.fulfill(
                content_type="text/html",
                body=_OPERATOR_HTML.replace(
                    "{sessionId:'session'}", "{sessionId:" + json.dumps(session_id) + "}"
                ),
            )
        else:
            result = await route.fetch(url=f"{root}{path}")
            await route.fulfill(response=result)

    context = await browser.new_context(
        extra_http_headers={"Authorization": "Basic b3BlcmF0b3I6cGFzc3dvcmQ="}
    )
    try:
        page = await context.new_page()
        from tests.server._browser_operator_keyboard import KeyboardOperator

        keyboard = KeyboardOperator(page) if keyboard_only else None

        async def activate(target):
            if keyboard is not None:
                await keyboard.activate(target)
            else:
                await target.click()

        page.set_default_timeout(5000)
        # Match the existing deterministic coordinator/guest fixture clock.
        if clock_ms is not None:
            await page.add_init_script(f"Date.now = () => {int(clock_ms)}")
        await page.route("https://operator.test/**", forward)
        await page.route_web_socket("wss://operator.test/api/browser-control/*", forward_socket)
        await page.goto(
            "https://operator.test/operator-test"
            if dashboard_path is None
            else f"https://operator.test{dashboard_path}/sessions/{session_id}"
        )
        await activate(page.get_by_role("button", name="Discover browsers"))
        await activate(page.get_by_role("button", name="Browser 1 · agent_controlled"))
        identity = owner.bound.record.identity
        await expect(
            page.get_by_text(f"Application purpose: {identity.operator_purpose.code}.", exact=True)
        ).to_be_visible()
        for origin in identity.operator_purpose.expected_origins:
            await expect(page.get_by_role("listitem").filter(has_text=origin)).to_be_visible()
        await expect(page.get_by_test_id("selected-browser-identity")).to_have_text(
            identity.browser_session_id
        )
        await expect(page.get_by_test_id("selected-browser-environment")).to_have_text(
            identity.environment_name
        )
        await expect(page.get_by_test_id("selected-browser-allocation")).to_have_text(
            identity.allocation_fingerprint
        )
        if expected_page_origin is not None:
            await expect(
                page.get_by_text(f"Page 1 observed origin: {expected_page_origin}.", exact=True)
            ).to_be_visible()
        if changed_page_origin is not None:
            # Simulate page-driven navigation in the actual native browser, not
            # a synthesized /pages response. Read-only refresh must disclose it
            # without silently refreshing input/takeover targets.
            assert daemon.active_page_id is not None
            native_page = daemon.pages[daemon.active_page_id].page
            await native_page.goto(changed_page_origin + "/next")
            await expect(
                page.get_by_text(f"Page 1 observed origin: {changed_page_origin}.", exact=True)
            ).to_be_visible()
            assert private_text not in await page.locator("body").inner_text()
            assert daemon.pages[daemon.active_page_id].revision is None
            assert not daemon.pages[daemon.active_page_id].refs
            # Explicit discovery is still required for new action authority.
            await activate(page.get_by_role("button", name="Browser 1 · agent_controlled"))
        try:
            await page.get_by_label("Profile checkpoint consent for the next takeover").wait_for()
        except Exception:
            raise AssertionError(responses) from None
        consent = page.get_by_label("Profile checkpoint consent for the next takeover")
        if keyboard is not None:
            await keyboard.deny_checkpoint(consent)
            await expect(
                page.get_by_role("status").filter(has_text="Control: agent_controlled")
            ).to_be_visible()
            await expect(
                page.get_by_role("status").filter(has_text="Page 1 observed origin:")
            ).to_be_visible()
        else:
            await consent.select_option("deny")
        # Exercise the ordinary view -> replace -> takeover journey, not only
        # viewers first opened after takeover. Every retired delivery must purge.
        phase = "view before takeover"
        await activate(page.get_by_role("button", name="View page 1", exact=True))
        await page.wait_for_function("() => document.querySelector('canvas').width > 0")
        phase = "replace viewer"
        await activate(page.get_by_role("button", name="View page 1", exact=True))
        await page.wait_for_function("() => document.querySelector('canvas').width > 0")
        phase = "takeover"
        await activate(page.get_by_role("button", name="Request exclusive takeover"))
        async with asyncio.timeout(5):
            while daemon.control.state != "operator_controlled":
                await asyncio.sleep(0.01)
        await activate(page.get_by_role("button", name="Refresh control state"))
        await expect(page.get_by_role("button", name="Prepare sensitive entry")).to_be_enabled()
        await activate(page.get_by_role("button", name="Browser 1 · operator_controlled"))
        canvas = page.get_by_label("Private live browser frame", exact=True)
        phase = "view after takeover"
        assert await canvas.evaluate("node => [node.width, node.height]") == [0, 0]
        await activate(page.get_by_role("button", name="View page 1", exact=True))
        await page.wait_for_function(
            "() => document.querySelector('canvas').width > 0 && document.querySelector('canvas').height > 0"
        )
        private_value = page.get_by_label("Private value", exact=True)
        await expect(private_value).to_have_count(0)
        purged.clear()
        hold_purge = True
        await activate(page.get_by_role("button", name="Prepare sensitive entry"))
        await asyncio.wait_for(purged.wait(), 5)
        _, pending = await owner.coordinator._load(owner.bound.record.identity)
        assert pending.sensitive_entry_pending
        assert not pending.sensitive_entry
        await expect(private_value).to_have_count(0)
        assert await canvas.evaluate("node => [node.width, node.height]") == [0, 0]
        release_purge.set()
        async with asyncio.timeout(5):
            while not owner.bound.record.sensitive_entry:
                await asyncio.sleep(0.01)
        assert daemon.control.capture_restricted
        assert not owner.bound.record.sensitive_entry_pending
        await activate(page.get_by_role("button", name="Refresh control state"))
        await expect(private_value).to_be_visible()
        send = page.get_by_role("button", name="Send private text once")
        await expect(send).to_be_disabled()
        await activate(page.get_by_role("button", name="Browser 1 · operator_controlled"))
        await expect(send).to_be_enabled()
        await activate(page.get_by_role("button", name="Next field", exact=True))
        try:
            async with asyncio.timeout(5):
                while owner.bound.record.settled_input_sequence != 1:
                    await asyncio.sleep(0.01)
        except TimeoutError:
            raise AssertionError((responses, protocol)) from None
        await expect(send).to_be_disabled()
        await activate(page.get_by_role("button", name="Browser 1 · operator_controlled"))
        await expect(send).to_be_enabled()
        if keyboard is not None:
            await keyboard.type_private(private_value, private_text)
        else:
            await private_value.fill(private_text)
        await activate(send)
        try:
            async with asyncio.timeout(5):
                while owner.bound.record.settled_input_sequence != 2:
                    await asyncio.sleep(0.01)
        except TimeoutError:
            raise AssertionError((responses, protocol)) from None
        await expect(private_value).to_have_value("")
        await expect(send).to_be_disabled()
        assert daemon.total_operations == initial_operations + 2
        assert token is not None
        assert owner.bound.record.request is not None
        assert owner.bound.record.checkpoint_consent == "deny"
        takeover = owner.bound.record.request
        await activate(page.get_by_role("button", name="Return control to agent"))
        await expect(private_value).to_have_count(0)
        async with asyncio.timeout(5):
            while owner.bound.record.state != "agent_controlled":
                await asyncio.sleep(0.01)
        assert daemon.control.fresh_observation_required
        if keyboard is not None:
            await activate(page.get_by_role("button", name="Refresh control state"))
            await expect(
                page.get_by_role("status").filter(has_text="Control: agent_controlled")
            ).to_be_visible()
        return token, takeover
    except BaseException as error:
        if failures is not None:
            failures.append(
                BaseExceptionGroup(
                    f"Rendered journey failed at {phase} after {keyboard.activations if keyboard else 0} keyboard activations",
                    [error],
                )
            )
        raise
    finally:
        release_purge.set()
        try:
            await context.close()
        finally:
            cancelled = set()
            for task in sockets:
                if not task.done() and task.cancel():
                    cancelled.add(task)
            settled = await asyncio.gather(*sockets, return_exceptions=True)
            cleanup_errors = [
                result
                for task, result in zip(sockets, settled, strict=True)
                if isinstance(result, BaseException)
                and not (
                    task in cancelled
                    and task.cancelled()
                    and isinstance(result, asyncio.CancelledError)
                )
            ]
            if cleanup_errors:
                raise BaseExceptionGroup("Rendered socket cleanup failed", cleanup_errors)
