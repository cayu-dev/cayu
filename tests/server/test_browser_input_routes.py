"""Private input socket admission refuses untrusted transport authority."""

import asyncio

import pytest
from fastapi import FastAPI
from tests.core.test_browser_control import operator_purpose
from tests.core.test_browser_control_authorization import Policy
from tests.core.test_browser_control_coordinator import coordinator, intent_for
from tests.core.test_browser_control_publisher import publication_fixture

from cayu.runtime._browser_control_input_tickets import BrowserInputTickets
from cayu.runtime._browser_control_publisher import BrowserControlPublisher
from cayu.runtime._browser_control_service import BrowserControlService
from cayu.runtime.browser_control import (
    BrowserControlPrincipal,
    BrowserSensitiveEntryIntent,
    BrowserTextInputIntent,
)
from cayu.server._browser_input_routes import (
    OPERATOR_INPUT_SUBPROTOCOL,
    create_browser_input_router,
)


@pytest.mark.parametrize(
    "mode",
    [
        "plaintext",
        "query",
        "origin",
        "protocol",
        "ticket",
        "binary",
        "cancel",
        "cancel_close_failure",
    ],
)
def test_input_socket_rejects_before_policy_or_dispatch(tmp_path, mode):
    async def scenario():
        async with publication_fixture("memory", tmp_path) as (store, _):
            policy = Policy(True)
            owner = coordinator(store, policy)
            service = BrowserControlService(
                purpose=operator_purpose(), guest_endpoint="wss://guest.test/control"
            )
            app = FastAPI()
            app.include_router(
                create_browser_input_router(
                    coordinator=owner,
                    service=service,
                    tickets=BrowserInputTickets(owner),
                    allowed_origin="https://operator.test",
                )
            )
            before = await store.load_checkpoint("session")
            incoming, outgoing = asyncio.Queue(), asyncio.Queue()
            await incoming.put({"type": "websocket.connect"})
            scope = {
                "type": "websocket",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "scheme": "ws" if mode == "plaintext" else "wss",
                "path": "/browser-control/input",
                "raw_path": b"/browser-control/input",
                "root_path": "",
                "query_string": b"token=untrusted" if mode == "query" else b"",
                "subprotocols": ["wrong"] if mode == "protocol" else [OPERATOR_INPUT_SUBPROTOCOL],
                "headers": [
                    (
                        b"origin",
                        b"https://wrong.test" if mode == "origin" else b"https://operator.test",
                    )
                ],
                "server": ("operator.test", 443),
                "client": ("127.0.0.1", 1234),
            }
            close_error = RuntimeError("close failed")

            async def send(message):
                await outgoing.put(message)
                if mode == "cancel_close_failure" and message["type"] == "websocket.close":
                    raise close_error

            task = asyncio.create_task(app(scope, incoming.get, send))
            async with asyncio.timeout(5):
                first = await outgoing.get()
                if mode in {"cancel", "cancel_close_failure"}:
                    assert first["type"] == "websocket.accept"
                    assert task.cancel()
                    with pytest.raises(asyncio.CancelledError) as raised:
                        await task
                    assert task.cancelled() and task.cancelling() == 1
                    if mode == "cancel_close_failure":
                        assert raised.value.__cause__ is close_error
                    first = await outgoing.get()
                if mode in {"ticket", "binary"}:
                    assert first["type"] == "websocket.accept"
                    await incoming.put(
                        {
                            "type": "websocket.receive",
                            **(
                                {"bytes": b"private-canary"}
                                if mode == "binary"
                                else {"text": "0" * 64}
                            ),
                        }
                    )
                    first = await outgoing.get()
                assert first["type"] == "websocket.close"
                assert "private-canary" not in repr(first)
                if not task.cancelled():
                    await task
            assert not policy.requests
            assert not service._owners
            assert await store.load_checkpoint("session") == before

    asyncio.run(scenario())


@pytest.mark.parametrize("payload", [b"", b"\xff", b"canary\x00", b"a" * 16385, b"a" * 4097])
def test_authenticated_input_rejects_malformed_payload_before_admission(
    tmp_path, monkeypatch, payload
):
    async def scenario():
        async with publication_fixture("memory", tmp_path) as (store, bootstrap):
            await BrowserControlPublisher(store).publish(bootstrap)
            owner = coordinator(store, Policy(True))
            principal = BrowserControlPrincipal(subject="operator")
            pending = await owner.request_takeover(
                principal=principal,
                operator_session_id="operator-session",
                intent=intent_for(bootstrap),
            )
            acquired = await owner._publish_guest_acquisition(expected=pending, lease_until_ms=4000)
            sensitive = await owner.request_sensitive_entry(
                principal=principal,
                operator_session_id="operator-session",
                intent=BrowserSensitiveEntryIntent(
                    identity=acquired.identity,
                    expected_record_revision=acquired.revision,
                    expected_control_epoch=acquired.control_epoch,
                    request_id=acquired.request.request_id,
                ),
            )
            ready = await owner._publish_guest_sensitive_entry(expected=sensitive)
            tickets = BrowserInputTickets(owner)
            token = await tickets.issue(
                principal=principal,
                operator_session_id="operator-session",
                intent=BrowserTextInputIntent(
                    identity=ready.identity,
                    expected_record_revision=ready.revision,
                    expected_control_epoch=ready.control_epoch,
                    request_id=ready.request.request_id,
                    input_sequence=1,
                    page=ready.request.pages[0],
                ),
            )
            service = BrowserControlService(
                purpose=operator_purpose(), guest_endpoint="wss://guest.test/control"
            )
            calls = []

            async def forbidden(**kwargs):
                calls.append(True)
                raise AssertionError("Malformed payload reached input admission")

            monkeypatch.setattr(service, "submit_text_input", forbidden)
            app = FastAPI()
            app.include_router(
                create_browser_input_router(
                    coordinator=owner,
                    service=service,
                    tickets=tickets,
                    allowed_origin="https://operator.test",
                )
            )
            incoming, outgoing = asyncio.Queue(), asyncio.Queue()
            before = await store.load_checkpoint("session")
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
            task = asyncio.create_task(app(scope, incoming.get, outgoing.put))
            async with asyncio.timeout(5):
                assert (await outgoing.get())["type"] == "websocket.accept"
                await incoming.put({"type": "websocket.receive", "text": token})
                assert (await outgoing.get())["text"] == "ready"
                await incoming.put({"type": "websocket.receive", "bytes": payload})
                response = await outgoing.get()
                assert response["type"] == "websocket.close"
                assert "canary" not in repr(response)
                await task
            assert not calls
            assert await store.load_checkpoint("session") == before
            assert not tickets._pending

    asyncio.run(scenario())
