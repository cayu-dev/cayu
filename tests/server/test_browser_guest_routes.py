"""Guest handshake admission through the actual ASGI routing boundary."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from tests.core.test_browser_control import operator_purpose
from tests.core.test_browser_control_authorization import Policy
from tests.core.test_browser_control_coordinator import coordinator, intent_for
from tests.core.test_browser_control_publisher import publication_fixture
from tests.core.test_browser_control_service import bound_browser_context
from tests.core.test_browser_session import _interactive_request

from cayu.runtime._browser_control_authorization import BrowserControlPermissionDenied
from cayu.runtime._browser_control_frames import BrowserViewUnavailable
from cayu.runtime._browser_control_input_tickets import BrowserInputTickets
from cayu.runtime._browser_control_service import BrowserControlService
from cayu.runtime.browser_control import (
    BrowserControlConflict,
    BrowserControlPage,
    BrowserControlPrincipal,
    BrowserHandbackIntent,
    BrowserSensitiveEntryIntent,
    BrowserTextInputIntent,
)
from cayu.server._browser_control_routes import (
    BrowserOperatorSessionTokens,
    create_browser_control_router,
)
from cayu.server._browser_guest_routes import create_browser_guest_router
from cayu.server._browser_input_routes import (
    OPERATOR_INPUT_SUBPROTOCOL,
    create_browser_input_router,
)
from cayu.server.auth import BasicAuth
from cayu.tools._browser_control_guest import GuestControlChannel, GuestControlFence
from cayu.tools._browser_control_transport import CONTROL_SUBPROTOCOL
from cayu.tools._browser_guest import _InteractiveDaemon, _InteractivePage
from cayu.tools.browser_session import BrowserSessionTool, _RunnerBrowserSessionBackend


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize(
    "lost_ack,takeover,view,handback",
    [
        (False, False, False, False),
        (True, False, False, False),
        (False, True, False, False),
        (False, False, True, False),
        (False, False, "profile", False),
        (False, True, False, True),
        (False, True, False, "cancel"),
        (False, "sensitive", False, False),
        (False, "sensitive_key", False, False),
        (False, "sensitive_cancel", False, False),
        (False, "sensitive_input_cancel", False, False),
        (False, "sensitive_input_loss", False, False),
    ],
)
def test_guest_asgi_binds_and_disconnects_with_real_guest_protocol(
    tmp_path, monkeypatch, backend, lost_ack, takeover, view, handback, malformed_disconnect=False
):
    async def scenario():
        async with publication_fixture(backend, tmp_path) as (store, bootstrap):
            exact = bootstrap.changed_record.identity
            service = BrowserControlService(
                purpose=operator_purpose(),
                guest_endpoint="wss://control.example/browser-control/guest",
            )
            private_delivery = asyncio.Queue()

            async def deliver(self, ctx, **kwargs):
                await private_delivery.put(kwargs)

            monkeypatch.setattr(_RunnerBrowserSessionBackend, "bootstrap_control", deliver)
            runner_backend = BrowserSessionTool()._backend
            assert type(runner_backend) is _RunnerBrowserSessionBackend
            bootstrap_task = asyncio.create_task(
                service.bootstrap(
                    bound_browser_context(exact),
                    backend=runner_backend,
                    browser_session_id=exact.browser_session_id,
                    arguments={"operation": "navigate"},
                )
            )
            material = await private_delivery.get()
            owner = coordinator(store, Policy(True))
            app = FastAPI()
            app.include_router(create_browser_guest_router(service=service, coordinator=owner))
            sessions = BrowserOperatorSessionTokens(b"k" * 32)
            input_tickets = BrowserInputTickets(owner)
            app.include_router(
                create_browser_control_router(
                    coordinator=owner,
                    auth=BasicAuth(username="operator", password="password"),
                    sessions=sessions,
                    allowed_origin="https://operator.test",
                    input_tickets=input_tickets,
                    service=service,
                )
            )
            app.include_router(
                create_browser_input_router(
                    coordinator=owner,
                    service=service,
                    tickets=input_tickets,
                    allowed_origin="https://operator.test",
                )
            )

            async def post_control(path, **kwargs):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app),
                    base_url="https://operator.test",
                    auth=httpx.BasicAuth("operator", "password"),
                ) as client:
                    return await client.post("/browser-control/" + path, **kwargs)

            incoming, outgoing = asyncio.Queue(), asyncio.Queue()
            await incoming.put({"type": "websocket.connect"})
            scope = {
                "type": "websocket",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "scheme": "wss",
                "path": "/browser-control/guest",
                "raw_path": b"/browser-control/guest",
                "root_path": "",
                "query_string": b"",
                "subprotocols": [CONTROL_SUBPROTOCOL],
                "headers": [
                    (b"authorization", ("Bearer " + material["credential"]).encode("ascii"))
                ],
                "server": ("control.example", 443),
                "client": ("127.0.0.1", 1234),
            }
            server = asyncio.create_task(app(scope, incoming.get, outgoing.put))
            assert (await outgoing.get())["type"] == "websocket.accept"
            status_seen = asyncio.Event()
            operator_seen = asyncio.Event()
            handback_seen = asyncio.Event()
            handback_replies = []

            class Connection:
                async def send(self, value):
                    if type(value) is bytes:
                        await incoming.put({"type": "websocket.receive", "bytes": value})
                        return
                    decoded = json.loads(value)
                    if lost_ack and json.loads(value).get("kind") == "bound":
                        await incoming.put({"type": "websocket.disconnect", "code": 1006})
                        raise ConnectionError("binding acknowledgement lost")
                    if json.loads(value).get("kind") == "settled":
                        if (
                            decoded.get("control_epoch") == 3
                            and decoded.get("state") == "agent_controlled"
                        ):
                            handback_replies.append(decoded)
                            if len(handback_replies) == 2:
                                handback_seen.set()
                        status_seen.set()
                        if (
                            decoded.get("state") == "operator_controlled"
                            and "request_id" not in decoded
                        ):
                            operator_seen.set()
                    await incoming.put({"type": "websocket.receive", "text": value})

                async def recv(self):
                    message = await outgoing.get()
                    if message["type"] == "websocket.close":
                        raise ConnectionError("server closed")
                    return message["text"]

                async def close(self):
                    await incoming.put({"type": "websocket.disconnect", "code": 1000})

            daemon = _InteractiveDaemon(exact.browser_session_id)
            daemon.context = object()
            daemon.configuration_limits = _interactive_request("observe").limits
            daemon.control = GuestControlFence(
                worker_instance=daemon.visual_worker_instance, wall_clock=lambda: 1.0
            )
            pixels = (
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
                + (16).to_bytes(4, "big") * 2
                + b"private-asgi-pixel-canary"
            )
            captures = []
            inputs = []
            input_entered, release_input = asyncio.Event(), asyncio.Event()

            async def insert_text(value):
                if takeover in {"sensitive_input_cancel", "sensitive_input_loss"}:
                    input_entered.set()
                    await release_input.wait()
                inputs.append(value)

            async def screenshot(**kwargs):
                assert kwargs["full_page"] is False
                captures.append(kwargs)
                return pixels

            daemon.pages["page"] = _InteractivePage(
                page=SimpleNamespace(
                    url="https://site.test/",
                    keyboard=SimpleNamespace(insert_text=insert_text, press=insert_text),
                    viewport_size={"width": 16, "height": 16},
                    screenshot=screenshot,
                ),
                session_id=exact.browser_session_id,
                page_id="page",
                revision="revision",
                lifecycle="active",
                configured=True,
            )
            daemon.active_page_id = "page"
            channel = asyncio.create_task(
                GuestControlChannel(daemon, scope_sha256=material["scope_sha256"]).run(Connection())
            )
            try:
                if not lost_ack:
                    async with asyncio.timeout(10):
                        bound = await bootstrap_task
                        await status_seen.wait()
                    assert bound.record.identity.worker_instance_id == daemon.visual_worker_instance
                    if view:
                        before = await store.load_checkpoint(exact.session_id)
                        continuity = await post_control("operator-session")
                        pages_response = await post_control(
                            "pages",
                            headers={
                                "X-Cayu-Browser-Operator": continuity.json()[
                                    "operator_session_token"
                                ]
                            },
                            json={
                                "identity": bound.record.identity.model_dump(mode="json"),
                                "expected_record_revision": bound.record.revision,
                            },
                        )
                        assert pages_response.status_code == 200
                        assert pages_response.headers["cache-control"] == "no-store"
                        assert pages_response.json() == {
                            "active_page_id": "page",
                            "pages": [
                                {"page_id": "page", "revision": "revision", "control_epoch": 1}
                            ],
                            "locations": [
                                {
                                    "page": {
                                        "page_id": "page",
                                        "revision": "revision",
                                        "control_epoch": 1,
                                    },
                                    "origin": "https://site.test",
                                }
                            ],
                        }
                        assert not captures and not inputs
                        page = BrowserControlPage(
                            page_id="page", revision="revision", control_epoch=1
                        )
                        if view == "profile":
                            daemon.profile_output_values = ("profile-secret-canary",)
                        capture = service.capture_view(
                            identity=bound.record.identity,
                            principal=BrowserControlPrincipal(subject="operator"),
                            operator_session_id="operator-session",
                            page=page,
                            until_ms=4000,
                        )
                        if view == "profile":
                            with pytest.raises(BrowserViewUnavailable):
                                await capture
                            assert not captures
                            status_seen.clear()
                            async with asyncio.timeout(2):
                                await status_seen.wait()
                            assert not server.done() and not channel.done()
                        else:
                            frame = await capture
                            assert frame.png == pixels and len(captures) == 1
                            assert "private-asgi-pixel-canary" not in repr(frame)
                        assert await store.load_checkpoint(exact.session_id) == before
                    if takeover:
                        operator_session = "operator-session"
                        token = None
                        if takeover in {
                            "sensitive",
                            "sensitive_key",
                            "sensitive_cancel",
                            "sensitive_input_cancel",
                            "sensitive_input_loss",
                        }:
                            response = await post_control("operator-session")
                            assert response.status_code == 200
                            token = response.json()["operator_session_token"]
                            operator_session = sessions.verify(
                                token, BrowserControlPrincipal(subject="operator")
                            )
                        pending = await owner.request_takeover(
                            principal=BrowserControlPrincipal(subject="operator"),
                            operator_session_id=operator_session,
                            intent=intent_for(bootstrap).model_copy(
                                update={"identity": bound.record.identity}
                            ),
                        )
                        assert pending.state == "takeover_requested"
                        async with asyncio.timeout(10):
                            await operator_seen.wait()
                        _, acquired = await owner._load(pending.identity)
                        assert acquired.state == "operator_controlled"
                        assert acquired.control_epoch == 2
                        if takeover in {
                            "sensitive",
                            "sensitive_key",
                            "sensitive_cancel",
                            "sensitive_input_cancel",
                            "sensitive_input_loss",
                        }:
                            viewer = service.register_viewer(acquired.identity)
                            viewer.begin_send()
                            viewer.finish_send()
                            response = await post_control(
                                "sensitive-entry",
                                headers={"X-Cayu-Browser-Operator": token},
                                json=BrowserSensitiveEntryIntent(
                                    identity=acquired.identity,
                                    expected_record_revision=acquired.revision,
                                    expected_control_epoch=acquired.control_epoch,
                                    request_id=acquired.request.request_id,
                                ).model_dump(mode="json"),
                            )
                            assert response.status_code == 202
                            assert response.json()["sensitive_entry_pending"] is True
                            assert response.json()["sensitive_entry"] is False
                            _, requested = await owner._load(acquired.identity)
                            assert requested.sensitive_entry_pending
                            async with asyncio.timeout(5):
                                while viewer.purge_token is None:
                                    await asyncio.sleep(0.01)
                            assert not daemon.control.sensitive_entry
                            assert (await owner._load(acquired.identity))[1] == requested
                            if takeover == "sensitive_cancel":
                                assert server.cancel()
                                with pytest.raises(asyncio.CancelledError):
                                    await server
                                assert server.cancelled() and server.cancelling() == 1
                                _, settled = await owner._load(acquired.identity)
                                assert settled.state == "control_uncertain"
                                assert settled.sensitive_entry_pending
                                assert not settled.sensitive_entry
                                assert not daemon.control.sensitive_entry
                                assert viewer in service._viewers
                                assert not viewer.purge_settled
                            else:
                                viewer.acknowledge_purge(viewer.purge_token)
                                async with asyncio.timeout(5):
                                    while True:
                                        _, settled = await owner._load(acquired.identity)
                                        if settled.sensitive_entry:
                                            break
                                        await asyncio.sleep(0.01)
                                assert not settled.sensitive_entry_pending
                                assert settled.capture_restricted and daemon.control.sensitive_entry
                                assert settled.revision == requested.revision + 1
                                assert settled.control_epoch == acquired.control_epoch
                                service.retire_viewer(viewer)
                                assert not service._viewers
                                commands = next(iter(service._owners.values())).commands
                                assert commands is not None
                                input_intent = BrowserTextInputIntent(
                                    identity=settled.identity,
                                    expected_record_revision=settled.revision,
                                    expected_control_epoch=settled.control_epoch,
                                    request_id=settled.request.request_id,
                                    input_sequence=1,
                                    page=settled.request.pages[0],
                                    input_kind="tab" if takeover == "sensitive_key" else "text",
                                )

                                async def submit_input():
                                    if takeover not in {"sensitive", "sensitive_key"}:
                                        return await service.submit_text_input(
                                            principal=BrowserControlPrincipal(subject="operator"),
                                            operator_session_id=operator_session,
                                            intent=input_intent,
                                            text="private-native-input-canary",
                                        )
                                    issued = await post_control(
                                        "input-ticket",
                                        headers={"X-Cayu-Browser-Operator": token},
                                        json=input_intent.model_dump(mode="json"),
                                    )
                                    assert issued.status_code == 200
                                    client_in, client_out = asyncio.Queue(), asyncio.Queue()
                                    await client_in.put({"type": "websocket.connect"})
                                    input_scope = {
                                        **scope,
                                        "path": "/browser-control/input",
                                        "raw_path": b"/browser-control/input",
                                        "headers": [(b"origin", b"https://operator.test")],
                                        "subprotocols": [OPERATOR_INPUT_SUBPROTOCOL],
                                    }
                                    input_socket = asyncio.create_task(
                                        app(input_scope, client_in.get, client_out.put)
                                    )
                                    assert (await client_out.get())["type"] == "websocket.accept"
                                    await client_in.put(
                                        {
                                            "type": "websocket.receive",
                                            "text": issued.json()["ticket"],
                                        }
                                    )
                                    assert (await client_out.get())["text"] == "ready"
                                    await client_in.put(
                                        {
                                            "type": "websocket.receive",
                                            "bytes": b"tab"
                                            if takeover == "sensitive_key"
                                            else b"private-native-input-canary",
                                        }
                                    )
                                    receipt = await client_out.get()
                                    assert "private-native-input-canary" not in repr(receipt)
                                    assert (
                                        json.loads(receipt["text"])["settled_input_sequence"] == 1
                                    )
                                    await input_socket
                                    with pytest.raises(BrowserControlPermissionDenied):
                                        input_tickets.consume(issued.json()["ticket"])
                                    return (await owner._load(settled.identity))[1]

                                input_task = asyncio.create_task(submit_input())
                                if takeover in {"sensitive_input_cancel", "sensitive_input_loss"}:
                                    async with asyncio.timeout(5):
                                        await input_entered.wait()
                                    _, in_flight = await owner._load(settled.identity)
                                    assert in_flight.pending_input_sequence == 1
                                    native_input = daemon._operator_input_task
                                    assert native_input is not None and not native_input.done()
                                    if takeover == "sensitive_input_loss":
                                        assert server.cancel()
                                        with pytest.raises(asyncio.CancelledError):
                                            await server
                                        assert server.cancelled() and server.cancelling() == 1
                                        with pytest.raises(BrowserControlConflict):
                                            await input_task
                                        assert not input_task.cancelled()
                                        _, input_result = await owner._load(settled.identity)
                                        assert input_result.state == "control_uncertain"
                                        assert input_result.pending_input_sequence == 1
                                        assert not native_input.done()
                                    else:
                                        assert input_task.cancel()
                                        with pytest.raises(asyncio.CancelledError):
                                            await input_task
                                        assert (
                                            input_task.cancelled() and input_task.cancelling() == 1
                                        )
                                        assert not server.done()
                                    release_input.set()
                                    await native_input
                                    async with asyncio.timeout(5):
                                        while True:
                                            _, input_result = await owner._load(settled.identity)
                                            if (
                                                takeover == "sensitive_input_loss"
                                                or input_result.settled_input_sequence == 1
                                            ):
                                                break
                                            await asyncio.sleep(0.01)
                                else:
                                    input_result = await input_task
                                assert inputs == (
                                    ["Tab"]
                                    if takeover == "sensitive_key"
                                    else ["private-native-input-canary"]
                                )
                                assert [
                                    (item.page_id, item.operations)
                                    for item in input_result.operator_page_operations
                                ] == [(input_intent.page.page_id, 1)]
                                if takeover == "sensitive_input_loss":
                                    assert input_result.settled_input_sequence == 0
                                    assert input_result.pending_input_sequence == 1
                                    assert input_result.pending_input_page is not None
                                    assert input_result.manual_mutation_uncertain
                                    with pytest.raises(BrowserControlConflict):
                                        await service.submit_text_input(
                                            principal=BrowserControlPrincipal(subject="operator"),
                                            operator_session_id=operator_session,
                                            intent=BrowserTextInputIntent(
                                                identity=settled.identity,
                                                expected_record_revision=input_result.revision,
                                                expected_control_epoch=settled.control_epoch,
                                                request_id=settled.request.request_id,
                                                input_sequence=1,
                                                page=settled.request.pages[0],
                                            ),
                                            text="replacement-must-not-dispatch",
                                        )
                                    assert inputs == ["private-native-input-canary"]
                                else:
                                    assert input_result.settled_input_sequence == 1
                                    assert input_result.pending_input_sequence is None
                                    assert input_result.pending_input_page is None
                                    assert not input_result.manual_mutation_uncertain
                                assert "private-native-input-canary" not in repr(
                                    await store.load_checkpoint(exact.session_id)
                                )
                        if handback:
                            assert pending.request is not None
                            intent = BrowserHandbackIntent(
                                identity=acquired.identity,
                                expected_record_revision=acquired.revision,
                                expected_control_epoch=acquired.control_epoch,
                                request_id=pending.request.request_id,
                            )
                            with pytest.raises(BrowserControlConflict):
                                await owner.request_handback(
                                    principal=BrowserControlPrincipal(subject="other"),
                                    operator_session_id="operator-session",
                                    intent=intent,
                                )
                            committed = asyncio.Event()
                            release_ack = asyncio.Event()
                            writes = []
                            original = store.publish_session_operation_guarded_with_store_time

                            async def paused_ack(*args, **kwargs):
                                result = await original(*args, **kwargs)
                                writes.append(kwargs["idempotency_key"])
                                if len(writes) == 2:
                                    committed.set()
                                    await release_ack.wait()
                                return result

                            if handback == "cancel":
                                monkeypatch.setattr(
                                    store,
                                    "publish_session_operation_guarded_with_store_time",
                                    paused_ack,
                                )
                            returning = await owner.request_handback(
                                principal=BrowserControlPrincipal(subject="operator"),
                                operator_session_id="operator-session",
                                intent=intent,
                            )
                            assert returning.state == "handback_pending"
                            if handback == "cancel":
                                async with asyncio.timeout(10):
                                    await committed.wait()
                                assert server.cancel("handback channel stopped")
                                release_ack.set()
                                with pytest.raises(asyncio.CancelledError):
                                    await server
                                assert server.cancelled() and server.cancelling() == 1
                                assert len(writes) == 3
                            else:
                                async with asyncio.timeout(10):
                                    await handback_seen.wait()
                            _, returned = await owner._load(acquired.identity)
                            assert (
                                returned.state
                                == (
                                    "control_uncertain"
                                    if handback == "cancel"
                                    else "agent_controlled"
                                )
                                and returned.fresh_observation_required
                            )
                            assert returned.lease_until_ms is None
                            assert daemon.pages["page"].revision is None
                            assert daemon.pages["page"].control_epoch == 2
                    await incoming.put(
                        {"type": "websocket.receive", "text": "malformed-control"}
                        if malformed_disconnect
                        else {"type": "websocket.disconnect", "code": 1000}
                    )
                async with asyncio.timeout(10):
                    if handback != "cancel" and takeover not in {
                        "sensitive_cancel",
                        "sensitive_input_loss",
                    }:
                        if malformed_disconnect:
                            with pytest.raises(ExceptionGroup):
                                await server
                            assert not service.channels._tasks
                            assert await service.channels.drain()
                        else:
                            await server
                if lost_ack:
                    assert not bootstrap_task.done()
                    assert bootstrap_task.cancel("unacknowledged bootstrap caller disconnected")
                    with pytest.raises(asyncio.CancelledError):
                        await bootstrap_task
                    assert bootstrap_task.cancelled() and bootstrap_task.cancelling() == 1
                fenced = await owner.bind_guest(
                    allocation=service._capabilities.allocation_for_invocation(
                        bound_browser_context(exact),
                        purpose=operator_purpose(),
                        browser_session_id=exact.browser_session_id,
                        arguments={"operation": "navigate"},
                    ),
                    worker_instance_id=daemon.visual_worker_instance,
                )
                assert fenced.state == "control_uncertain"
                assert fenced.control_epoch == (3 if handback else 2 if takeover else 1)
                if view:
                    with pytest.raises(BrowserControlConflict):
                        await service.capture_view(
                            identity=bound.record.identity,
                            principal=BrowserControlPrincipal(subject="operator"),
                            operator_session_id="operator-session",
                            page=page,
                            until_ms=4000,
                        )
                    assert len(captures) == (0 if view == "profile" else 1)
            finally:
                await asyncio.gather(channel, return_exceptions=True)
                assert await service.drain_bootstrap_deliveries()
                assert await owner.drain()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_protocol_failure_with_settled_cleanup_retires_channel(tmp_path, monkeypatch, backend):
    test_guest_asgi_binds_and_disconnects_with_real_guest_protocol(
        tmp_path, monkeypatch, backend, False, False, False, False, malformed_disconnect=True
    )


@pytest.mark.parametrize(
    "case", ["invalid_token", "operator_auth", "origin", "query", "plaintext", "duplicate_auth"]
)
def test_guest_route_refuses_untrusted_admission_before_acceptance(tmp_path, case):
    async def scenario():
        async with publication_fixture("memory", tmp_path) as (store, _):
            before = await store.load_checkpoint("session")
            policy = Policy(True)
            app = FastAPI()
            app.include_router(
                create_browser_guest_router(
                    service=BrowserControlService(
                        purpose=operator_purpose(),
                        guest_endpoint="wss://control.example/browser-control/guest",
                    ),
                    coordinator=coordinator(store, policy),
                )
            )
            headers = [(b"authorization", b"Bearer " + b"a" * 64)]
            if case == "operator_auth":
                headers = [(b"authorization", b"Basic b3BlcmF0b3I6cGFzc3dvcmQ=")]
            elif case == "origin":
                headers.append((b"origin", b"https://operator.example"))
            elif case == "duplicate_auth":
                headers += headers
            scope = {
                "type": "websocket",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "scheme": "ws" if case == "plaintext" else "wss",
                "path": "/browser-control/guest",
                "raw_path": b"/browser-control/guest",
                "root_path": "",
                "query_string": b"token=secret" if case == "query" else b"",
                "headers": headers,
                "subprotocols": [CONTROL_SUBPROTOCOL],
                "server": ("control.example", 443),
                "client": ("127.0.0.1", 1234),
            }
            sent = []

            async def receive():
                return {"type": "websocket.connect"}

            async def send(message):
                sent.append(message)

            await app(scope, receive, send)
            assert len(sent) == 1
            assert sent[0]["type"] == "websocket.close"
            assert sent[0]["code"] == 1008
            assert not policy.requests
            assert await store.load_checkpoint("session") == before

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", [None, OSError, RuntimeError, asyncio.CancelledError])
def test_guest_socket_close_distinguishes_disconnect_from_cleanup_failure(failure):
    from starlette.websockets import WebSocket, WebSocketState

    from cayu.server._browser_guest_routes import _GuestSocket

    async def scenario():
        sent = []

        async def receive():
            return {"type": "websocket.connect"}

        async def send(message):
            sent.append(message)
            if message["type"] == "websocket.close" and failure is not None:
                raise failure()

        socket = WebSocket({"type": "websocket"}, receive, send)
        await socket.accept()
        connection = _GuestSocket(socket)
        if failure in {RuntimeError, asyncio.CancelledError}:
            with pytest.raises(failure):
                await connection.close()
        else:
            # Starlette turns ASGI OSError into WebSocketDisconnect(1006).
            await connection.close()
        assert socket.application_state is WebSocketState.DISCONNECTED
        assert sent[-1]["type"] == "websocket.close"

    asyncio.run(scenario())
