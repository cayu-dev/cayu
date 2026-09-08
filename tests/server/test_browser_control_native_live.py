"""HTTP control admission through TLS guest framing into isolated Chromium.

Operator HTTP/WebSocket use ASGI transports; the guest channel uses real TLS.
Runtime allocation bootstrap remains a separate acceptance row.
"""

import asyncio
import json
import os
from dataclasses import replace

import httpx
import pytest
from fastapi import FastAPI, WebSocket
from tests.core.test_browser_control import operator_purpose
from tests.core.test_browser_control_authorization import Policy
from tests.core.test_browser_control_coordinator import coordinator, intent_for
from tests.core.test_browser_control_publisher import publication_fixture
from tests.core.test_browser_control_transport import control_tls as _control_tls
from tests.core.test_browser_session import _interactive_request
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed
from websockets.typing import Subprotocol

from cayu.runtime._browser_control_channel import (
    BrowserGuestCommandOwner,
    bind_browser_guest_channel,
    browser_allocation_digest,
)
from cayu.runtime._browser_control_checkpoint import browser_control_checkpoint_read_scope
from cayu.runtime._browser_control_input_tickets import BrowserInputTickets
from cayu.runtime._browser_control_model import browser_model_control_admission
from cayu.runtime._browser_control_service import BrowserControlService, _BootstrapOwner
from cayu.runtime._browser_control_view_tickets import BrowserViewTickets
from cayu.runtime.browser_control import (
    BrowserControlAllocation,
    BrowserControlConflict,
)
from cayu.server._browser_control_routes import (
    BrowserOperatorSessionTokens,
    create_browser_control_router,
)
from cayu.server._browser_guest_routes import _GuestSocket, create_browser_guest_router
from cayu.server._browser_input_routes import (
    OPERATOR_INPUT_SUBPROTOCOL,
    create_browser_input_router,
)
from cayu.server._browser_viewer_routes import create_browser_viewer_router
from cayu.server.auth import BasicAuth
from cayu.tools._browser_control_guest import GuestControlChannel, GuestControlFence
from cayu.tools._browser_control_transport import CONTROL_SUBPROTOCOL, open_guest_control_channel
from cayu.tools._browser_guest import _GuestFailure, _InteractiveDaemon, _InteractivePage
from cayu.tools.browser_session import BrowserSessionTool, _RunnerBrowserSessionBackend

control_tls = _control_tls
pytestmark = pytest.mark.skipif(
    os.environ.get("CAYU_BROWSER_CONTROL_LIVE") != "1",
    reason="Opt-in isolated Chromium acceptance.",
)


async def _operator_input(api, owner, ticket, payload, *, consumed=False):
    incoming, outgoing = asyncio.Queue(), asyncio.Queue()
    await incoming.put({"type": "websocket.connect"})
    scope = {
        "type": "websocket",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "scheme": "wss",
        "path": "/browser-control/input",
        "root_path": "",
        "query_string": b"",
        "subprotocols": [OPERATOR_INPUT_SUBPROTOCOL],
        "headers": [(b"origin", b"https://operator.test")],
        "server": ("operator.test", 443),
        "client": ("127.0.0.1", 1234),
    }
    task = asyncio.create_task(api(scope, incoming.get, outgoing.put))
    try:
        async with asyncio.timeout(10):
            assert (await outgoing.get())["type"] == "websocket.accept"
            await incoming.put({"type": "websocket.receive", "text": ticket})
            admission = await outgoing.get()
            if consumed:
                assert admission["type"] == "websocket.close"
                await task
                return None
            assert admission["text"] == "ready"
            await incoming.put({"type": "websocket.receive", "bytes": payload})
            while owner._input is None:
                if task.done():
                    await task
                    pytest.fail("Input route ended before native admission")
                await asyncio.sleep(0)
            await owner.step()
            message = await outgoing.get()
            result = json.loads(message["text"])
            assert result["state"] == "settled"
            assert (await outgoing.get())["type"] == "websocket.close"
            await task
            return result
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.skipif(not os.environ.get("CAYU_BROWSER_DASHBOARD_URL"), reason="Requires local Vite.")
def test_rendered_operator_input_and_handback_reach_native_browser(tmp_path, control_tls, backend):
    test_http_takeover_tls_sensitive_input_and_handback(
        tmp_path, control_tls, backend, rendered=True
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_native_disconnect_before_input_acknowledgement(tmp_path, control_tls, backend):
    test_http_takeover_tls_sensitive_input_and_handback(
        tmp_path, control_tls, backend, mid_input_disconnect=True
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_native_tls_disconnect_keeps_both_input_owners_fenced(tmp_path, control_tls, backend):
    test_http_takeover_tls_sensitive_input_and_handback(
        tmp_path, control_tls, backend, disconnect=True
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_http_takeover_tls_sensitive_input_and_handback(
    tmp_path, control_tls, backend, disconnect=False, mid_input_disconnect=False, rendered=False
):
    from playwright.async_api import async_playwright

    async def scenario():
        async with publication_fixture(backend, tmp_path) as (store, bootstrap):
            control = coordinator(store, Policy(True))
            allocation = BrowserControlAllocation.model_validate(
                bootstrap.changed_record.identity.model_dump(exclude={"worker_instance_id"})
            )
            async with async_playwright() as playwright:
                browser = await playwright.chromium.launch(
                    executable_path=os.environ.get("CAYU_BROWSER_CONTROL_CHROMIUM"), headless=True
                )
                try:
                    context = await browser.new_context()
                    await context.route(
                        "https://manual.test/**",
                        lambda route: route.fulfill(
                            body="<label>Password <input id='password' type='password'></label>",
                            content_type="text/html",
                        ),
                    )
                    page = await context.new_page()
                    await page.goto("https://manual.test/")
                    daemon = _InteractiveDaemon(allocation.browser_session_id)
                    daemon.context = context
                    daemon.control = GuestControlFence(
                        worker_instance=daemon.visual_worker_instance, wall_clock=lambda: 1.0
                    )
                    daemon.configuration_limits = _interactive_request("observe").limits
                    daemon.pages["page"] = _InteractivePage(
                        page=page,
                        session_id=daemon.session_id,
                        page_id="page",
                        lifecycle="active",
                        revision="revision",
                        configured=True,
                    )
                    daemon.pages["page"].cdp = await context.new_cdp_session(page)
                    daemon.active_page_id = "page"
                    ready = asyncio.get_running_loop().create_future()
                    stop = asyncio.Event()
                    service = BrowserControlService(
                        purpose=operator_purpose(), guest_endpoint="wss://guest.test/control"
                    )

                    async def peer(connection):
                        try:
                            connected = False

                            async def receive():
                                nonlocal connected
                                if not connected:
                                    connected = True
                                    return {"type": "websocket.connect"}
                                try:
                                    value = await connection.recv()
                                except ConnectionClosed:
                                    return {"type": "websocket.disconnect", "code": 1000}
                                return {
                                    "type": "websocket.receive",
                                    "text" if isinstance(value, str) else "bytes": value,
                                }

                            async def send(message):
                                if message["type"] == "websocket.send":
                                    await connection.send(message.get("text", message.get("bytes")))
                                elif message["type"] == "websocket.close":
                                    await connection.close()

                            if rendered:
                                guest_api = FastAPI()
                                guest_api.include_router(
                                    create_browser_guest_router(
                                        service=service, coordinator=control
                                    )
                                )
                                assert connection.request is not None
                                scope = {
                                    "type": "websocket",
                                    "asgi": {"version": "3.0"},
                                    "http_version": "1.1",
                                    "scheme": "wss",
                                    "path": "/browser-control/guest",
                                    "root_path": "",
                                    "query_string": b"",
                                    "subprotocols": [connection.subprotocol],
                                    "headers": [
                                        (key.lower().encode(), value.encode())
                                        for key, value in connection.request.headers.raw_items()
                                    ],
                                    "server": ("127.0.0.1", 443),
                                    "client": ("127.0.0.1", 1234),
                                }
                                route_task = asyncio.create_task(guest_api(scope, receive, send))
                                try:
                                    pending_owner = service._owners[
                                        browser_allocation_digest(allocation)
                                    ]
                                    while pending_owner.commands is None:
                                        if route_task.done():
                                            await route_task
                                            raise AssertionError("Guest route ended before binding")
                                        await asyncio.sleep(0)
                                    ready.set_result(pending_owner.commands)
                                    await route_task
                                finally:
                                    if not route_task.done():
                                        route_task.cancel()
                                    await asyncio.gather(route_task, return_exceptions=True)
                                return
                            socket = WebSocket({"type": "websocket"}, receive, send)
                            await socket.accept()
                            framed = _GuestSocket(socket)
                            bound = await bind_browser_guest_channel(
                                coordinator=control, allocation=allocation, connection=framed
                            )
                            ready.set_result(
                                BrowserGuestCommandOwner(
                                    coordinator=control,
                                    connection=framed,
                                    bound=bound,
                                    suspend_viewer_delivery=service.suspend_viewer_delivery,
                                )
                            )
                            await stop.wait()
                        except Exception as error:
                            if not ready.done():
                                ready.set_exception(error)
                            else:
                                raise

                    server_tls, client_tls = control_tls
                    async with serve(
                        peer,
                        "127.0.0.1",
                        0,
                        ssl=server_tls,
                        subprotocols=[Subprotocol(CONTROL_SUBPROTOCOL)],
                    ) as server:
                        endpoint = f"wss://127.0.0.1:{server.sockets[0].getsockname()[1]}/guest"
                        if rendered:
                            from tests.server._native_control_bootstrap import (
                                bootstrap_native_control,
                            )

                            service = BrowserControlService(
                                purpose=operator_purpose(), guest_endpoint=endpoint
                            )
                            connection, guest = await bootstrap_native_control(
                                service, daemon, bootstrap.changed_record.identity, client_tls
                            )
                        else:
                            connection = await open_guest_control_channel(
                                endpoint=endpoint, credential="a" * 64, tls=client_tls
                            )
                            guest = asyncio.create_task(
                                GuestControlChannel(
                                    daemon, scope_sha256=browser_allocation_digest(allocation)
                                ).run(connection)
                            )
                        try:
                            async with asyncio.timeout(30):
                                owner = await ready
                                # Supply the completed bootstrap fixture, then use
                                # the production attachment and service lookup.
                                bound = asyncio.get_running_loop().create_future()
                                bound.set_result(owner.bound)

                                async def delivered():
                                    return ()

                                delivery = asyncio.create_task(delivered())
                                await delivery
                                runner_backend = BrowserSessionTool()._backend
                                assert isinstance(runner_backend, _RunnerBrowserSessionBackend)
                                if not rendered:
                                    service._owners[browser_allocation_digest(allocation)] = (
                                        _BootstrapOwner(allocation, runner_backend, bound, delivery)
                                    )
                                    service.attach_commands(owner)
                                sessions = BrowserOperatorSessionTokens(b"k" * 32)
                                tickets = BrowserInputTickets(control)
                                view_tickets = BrowserViewTickets(control)
                                api = FastAPI()
                                api.include_router(
                                    create_browser_control_router(
                                        coordinator=control,
                                        auth=BasicAuth(
                                            username="operator",
                                            password="password",
                                            tenant="tenant",
                                        ),
                                        sessions=sessions,
                                        allowed_origin="https://operator.test",
                                        input_tickets=tickets,
                                        view_tickets=view_tickets,
                                        service=service,
                                    )
                                )
                                api.include_router(
                                    create_browser_viewer_router(
                                        coordinator=control,
                                        service=service,
                                        tickets=view_tickets,
                                        allowed_origin="https://operator.test",
                                    )
                                )
                                api.include_router(
                                    create_browser_input_router(
                                        coordinator=control,
                                        service=service,
                                        tickets=tickets,
                                        allowed_origin="https://operator.test",
                                    )
                                )
                                async with httpx.AsyncClient(
                                    transport=httpx.ASGITransport(app=api),
                                    base_url="https://operator.test",
                                    auth=httpx.BasicAuth("operator", "password"),
                                ) as client:
                                    if rendered:
                                        from tests.server._browser_operator_rendering import (
                                            rendered_native_operator_journey,
                                        )

                                        token, takeover = await rendered_native_operator_journey(
                                            browser, client, api, owner, daemon
                                        )
                                    else:
                                        token = (
                                            await client.post("/browser-control/operator-session")
                                        ).json()["operator_session_token"]
                                        takeover = intent_for(bootstrap).model_copy(
                                            update={"identity": owner.bound.record.identity}
                                        )
                                        result = await client.post(
                                            "/browser-control/takeover",
                                            headers={"X-Cayu-Browser-Operator": token},
                                            json=takeover.model_dump(mode="json"),
                                        )
                                        assert result.status_code == 200, result.text
                                        assert daemon.control.state == "agent_controlled"
                                        await owner.step()
                                    headers = {"X-Cayu-Browser-Operator": token}
                                    assert daemon.control.state == (
                                        "agent_controlled" if rendered else "operator_controlled"
                                    )

                                    def intent():
                                        record = owner.bound.record
                                        return {
                                            "identity": record.identity.model_dump(mode="json"),
                                            "expected_record_revision": record.revision,
                                            "expected_control_epoch": record.control_epoch,
                                            "request_id": takeover.request_id,
                                        }

                                    if not rendered:
                                        response = await client.post(
                                            "/browser-control/sensitive-entry",
                                            headers=headers,
                                            json=intent(),
                                        )
                                        assert response.status_code == 202, response.text
                                        await owner.step()
                                    assert daemon.control.capture_restricted
                                    inputs = (
                                        []
                                        if rendered
                                        else [
                                            (1, "tab", "tab"),
                                            (2, "text", "tls-native-canary"),
                                        ]
                                    )
                                    for sequence, kind, text in inputs:
                                        response = await client.post(
                                            "/browser-control/input-ticket",
                                            headers=headers,
                                            json={
                                                **intent(),
                                                "input_sequence": sequence,
                                                "page": takeover.pages[0].model_dump(mode="json"),
                                                "input_kind": kind,
                                            },
                                        )
                                        assert response.status_code == 200, response.text
                                        if mid_input_disconnect and kind == "text":
                                            entered, release_native = (
                                                asyncio.Event(),
                                                asyncio.Event(),
                                            )
                                            keyboard = page.keyboard
                                            insert_text = keyboard.insert_text

                                            async def delayed_ack(
                                                value,
                                                insert_text=insert_text,
                                                entered=entered,
                                                release_native=release_native,
                                            ):
                                                await insert_text(value)
                                                entered.set()
                                                await release_native.wait()

                                            with pytest.MonkeyPatch.context() as patch:
                                                patch.setattr(keyboard, "insert_text", delayed_ack)
                                                sending = asyncio.create_task(
                                                    _operator_input(
                                                        api,
                                                        owner,
                                                        response.json()["ticket"],
                                                        text.encode("utf-8"),
                                                    )
                                                )
                                                try:
                                                    await asyncio.wait_for(entered.wait(), 5)
                                                    assert (
                                                        await page.locator(
                                                            "#password"
                                                        ).input_value()
                                                        == text
                                                    )
                                                    native = daemon._operator_input_task
                                                    assert native is not None and not native.done()
                                                    await connection.close()
                                                    await owner.disconnect()
                                                    _, fenced = await control._load(
                                                        owner.bound.record.identity
                                                    )
                                                    assert fenced.state == "control_uncertain"
                                                    assert fenced.pending_input_sequence == 2
                                                    assert fenced.settled_input_sequence == 1
                                                    assert not native.done()
                                                    with browser_control_checkpoint_read_scope(
                                                        fenced.identity.session_id
                                                    ):
                                                        checkpoint = await store.load_checkpoint(
                                                            fenced.identity.session_id
                                                        )
                                                    with pytest.raises(BrowserControlConflict):
                                                        browser_model_control_admission(
                                                            checkpoint,
                                                            allocation=allocation,
                                                            operation_name="observe",
                                                        )
                                                finally:
                                                    release_native.set()
                                                    await asyncio.wait_for(
                                                        asyncio.gather(
                                                            sending, guest, return_exceptions=True
                                                        ),
                                                        10,
                                                    )
                                            assert daemon.control.state == "control_uncertain"
                                            assert daemon.total_operations == 2
                                            _, retained = await control._load(fenced.identity)
                                            assert retained.pending_input_sequence == 2
                                            assert retained.settled_input_sequence == 1
                                            assert await control.drain()
                                            return
                                        settled = await _operator_input(
                                            api,
                                            owner,
                                            response.json()["ticket"],
                                            text.encode("utf-8"),
                                        )
                                        assert settled is not None
                                        assert settled["settled_input_sequence"] == sequence
                                        assert [
                                            (item.page_id, item.operations)
                                            for item in owner.bound.record.operator_page_operations
                                        ] == [("page", sequence)]
                                        assert daemon.total_operations == sequence
                                        assert daemon.pages["page"].operation_count == sequence
                                        assert not tickets._pending
                                        assert "tls-native-canary" not in repr(settled)
                                        before_replay = owner.bound.record
                                        assert (
                                            await _operator_input(
                                                api,
                                                owner,
                                                response.json()["ticket"],
                                                b"must-not-arrive",
                                                consumed=True,
                                            )
                                            is None
                                        )
                                        assert owner.bound.record == before_replay
                                    assert (
                                        await page.locator("#password").input_value()
                                        == "tls-native-canary"
                                    )
                                    if disconnect:
                                        previous = owner.bound.record
                                        await connection.close()
                                        await asyncio.wait_for(
                                            asyncio.gather(guest, return_exceptions=True), 5
                                        )
                                        assert daemon.control.state == "control_uncertain"
                                        await owner.disconnect()
                                        _, fenced = await control._load(previous.identity)
                                        assert fenced.state == "control_uncertain"
                                        assert fenced.operator_page_operations == (
                                            previous.operator_page_operations
                                        )
                                        denied = await client.post(
                                            "/browser-control/input-ticket",
                                            headers=headers,
                                            json={
                                                **intent(),
                                                "input_sequence": 3,
                                                "page": takeover.pages[0].model_dump(mode="json"),
                                                "input_kind": "text",
                                            },
                                        )
                                        assert denied.status_code == 409, denied.text
                                        assert not tickets._pending
                                        with browser_control_checkpoint_read_scope(
                                            fenced.identity.session_id
                                        ):
                                            checkpoint = await store.load_checkpoint(
                                                fenced.identity.session_id
                                            )
                                        with pytest.raises(BrowserControlConflict):
                                            browser_model_control_admission(
                                                checkpoint,
                                                allocation=allocation,
                                                operation_name="observe",
                                            )
                                        request = replace(
                                            _interactive_request("observe"),
                                            session_id=daemon.session_id,
                                            page_id="page",
                                            invocation_control_epoch=previous.control_epoch,
                                        )
                                        with pytest.raises(_GuestFailure):
                                            await daemon.execute(request)
                                        assert daemon.total_operations == 2
                                        assert await control.drain()
                                        return
                                    if not rendered:
                                        response = await client.post(
                                            "/browser-control/handback",
                                            headers=headers,
                                            json=intent(),
                                        )
                                        assert response.status_code == 200, response.text
                                        await owner.step()
                                    _, durable = await control._load(owner.bound.record.identity)
                                    assert durable.state == "agent_controlled"
                                    assert durable.fresh_observation_required
                                    assert durable.acquisition_audit is not None
                                    assert durable.handback_audit is not None
                                    assert durable.acquisition_audit.phase == "acquired"
                                    assert durable.handback_audit.phase == "handed_back"
                                    for audit in (
                                        durable.acquisition_audit,
                                        durable.handback_audit,
                                    ):
                                        assert audit.request_id == takeover.request_id
                                        assert tuple(item.origin for item in audit.locations) == (
                                            "https://manual.test",
                                        )
                                        assert "tls-native-canary" not in audit.model_dump_json()
                                    assert daemon.control.fresh_observation_required
                                    with browser_control_checkpoint_read_scope(
                                        durable.identity.session_id
                                    ):
                                        checkpoint = await store.load_checkpoint(
                                            durable.identity.session_id
                                        )
                                    admission = browser_model_control_admission(
                                        checkpoint,
                                        allocation=allocation,
                                        operation_name="observe",
                                    )
                                    assert admission is not None
                                    assert admission.operator_page_operations == (("page", 2),)
                                    assert admission.control_epoch == durable.control_epoch
                                    request = replace(
                                        _interactive_request("observe"),
                                        session_id=daemon.session_id,
                                        page_id="page",
                                        invocation_control_epoch=durable.control_epoch,
                                    )
                                    with pytest.raises(_GuestFailure):
                                        await daemon.execute(replace(request, operation="click"))
                                    observed = await daemon.execute(request)
                                    assert observed["kind"] == "success", observed
                                    assert observed["profile_output_protected"] is True
                                    assert observed["page_set"]["total_operations"] == 3
                                    assert observed["page_set"]["pages"][0]["operation_count"] == 3
                                    assert "tls-native-canary" not in repr(observed)
                                    assert not daemon.control.fresh_observation_required
                                    # Only runtime terminal publication may release the durable gate.
                                    _, durable = await control._load(durable.identity)
                                    assert durable.fresh_observation_required
                        finally:
                            stop.set()
                            await connection.close()
                            await asyncio.gather(guest, return_exceptions=True)
                    if rendered:
                        pending_owner = service._owners[browser_allocation_digest(allocation)]
                        assert pending_owner.commands is None
                        assert await service.channels.drain()
                        _, disconnected = await control._load(owner.bound.record.identity)
                        assert disconnected.state == "control_uncertain"
                    assert await control.drain()
                finally:
                    await browser.close()

    asyncio.run(scenario())
