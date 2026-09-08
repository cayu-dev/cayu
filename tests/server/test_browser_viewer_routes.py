"""Authenticated HTTP ticket to browser-compatible viewer WebSocket."""

import asyncio

import httpx
import pytest
from fastapi import FastAPI
from tests.core.test_browser_control import operator_purpose
from tests.core.test_browser_control_authorization import Policy
from tests.core.test_browser_control_coordinator import coordinator, intent_for
from tests.core.test_browser_control_publisher import publication_fixture

from cayu.runtime._browser_control_authorization import BrowserControlRevisionChanged
from cayu.runtime._browser_control_frames import BrowserViewUnavailable, PrivateBrowserFrame
from cayu.runtime._browser_control_publisher import BrowserControlPublisher
from cayu.runtime._browser_control_service import BrowserControlService
from cayu.runtime._browser_control_view_tickets import BrowserViewTickets
from cayu.runtime.browser_control import BrowserControlPrincipal
from cayu.server._browser_control_routes import (
    BrowserOperatorSessionTokens,
    create_browser_control_router,
)
from cayu.server._browser_viewer_routes import (
    OPERATOR_VIEW_SUBPROTOCOL,
    create_browser_viewer_router,
)
from cayu.server.auth import BasicAuth


@pytest.mark.parametrize(
    "mode",
    [
        "valid",
        "wrong_ticket",
        "wrong_origin",
        "revoked",
        "purge",
        "cancel_send",
        "wrong_purge",
        "empty_purge",
        "capture_purge",
        "purge_disconnect",
        "takeover_purge",
        "retire",
        "takeover_during_capture",
    ],
)
@pytest.mark.parametrize("stale_error", [BrowserViewUnavailable, BrowserControlRevisionChanged])
def test_http_ticket_viewer_handoff(tmp_path, monkeypatch, mode, stale_error):
    async def scenario():
        async with publication_fixture("sqlite", tmp_path) as (store, bootstrap):
            record = await BrowserControlPublisher(store).publish(bootstrap)
            policy = Policy(True)
            owner = coordinator(store, policy)
            tickets = BrowserViewTickets(owner)
            service = BrowserControlService(
                purpose=operator_purpose(), guest_endpoint="wss://guest.test/control"
            )
            app = FastAPI()
            app.include_router(
                create_browser_control_router(
                    coordinator=owner,
                    auth=BasicAuth(username="operator", password="password"),
                    sessions=BrowserOperatorSessionTokens(b"k" * 32),
                    allowed_origin="https://operator.test",
                    view_tickets=tickets,
                )
            )
            app.include_router(
                create_browser_viewer_router(
                    coordinator=owner,
                    service=service,
                    tickets=tickets,
                    allowed_origin="https://operator.test",
                )
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="https://operator.test",
                auth=httpx.BasicAuth("operator", "password"),
            ) as client:
                continuity = (await client.post("/browser-control/operator-session")).json()[
                    "operator_session_token"
                ]
                response = await client.post(
                    "/browser-control/view-ticket",
                    headers={"X-Cayu-Browser-Operator": continuity},
                    json={
                        "identity": record.identity.model_dump(mode="json"),
                        "expected_record_revision": 1,
                        "page": {"page_id": "page", "revision": "revision", "control_epoch": 1},
                    },
                )
                assert response.status_code == 200
                ticket = response.json()["ticket"]
            if mode == "revoked":
                policy.allowed = False
            captures = []
            capture_entered, release_capture = asyncio.Event(), asyncio.Event()

            async def capture(**kwargs):
                captures.append(kwargs)
                if mode == "takeover_during_capture" and len(captures) == 2:
                    capture_entered.set()
                    await release_capture.wait()
                    raise stale_error("Browser view changed before delivery.")
                if mode == "capture_purge":
                    capture_entered.set()
                    await release_capture.wait()
                return PrivateBrowserFrame(kwargs["page"], 1, 16, 16, b"private-pixels")

            monkeypatch.setattr(service, "capture_view", capture)
            incoming, outgoing = asyncio.Queue(), asyncio.Queue()
            await incoming.put({"type": "websocket.connect"})
            scope = {
                "type": "websocket",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "scheme": "wss",
                "path": "/browser-control/viewer",
                "raw_path": b"/browser-control/viewer",
                "root_path": "",
                "query_string": b"",
                "subprotocols": [OPERATOR_VIEW_SUBPROTOCOL],
                "headers": [
                    (
                        b"origin",
                        b"https://wrong.test"
                        if mode == "wrong_origin"
                        else b"https://operator.test",
                    )
                ],
                "server": ("operator.test", 443),
                "client": ("127.0.0.1", 1234),
            }
            send_entered = asyncio.Event()

            async def send(message):
                await outgoing.put(message)
                if mode == "cancel_send" and "bytes" in message:
                    send_entered.set()
                    await asyncio.Event().wait()

            task = asyncio.create_task(app(scope, incoming.get, send))
            first = await outgoing.get()
            if mode == "wrong_origin":
                assert first["type"] == "websocket.close"
            else:
                assert first["type"] == "websocket.accept"
                await incoming.put(
                    {
                        "type": "websocket.receive",
                        "text": "0" * 64 if mode == "wrong_ticket" else ticket,
                    }
                )
                reply = await outgoing.get()
                if mode in {"wrong_ticket", "revoked"}:
                    assert reply["type"] == "websocket.close"
                else:
                    assert reply["text"] == "ready"
                    if mode != "empty_purge":
                        await incoming.put({"type": "websocket.receive", "text": "frame"})
                        if mode != "capture_purge":
                            assert (await outgoing.get())["bytes"] == b"private-pixels"
                    if mode in {"empty_purge", "capture_purge"}:
                        if mode == "capture_purge":
                            await capture_entered.wait()
                        await service.suspend_viewer_delivery(record.identity)
                        delivery = next(iter(service._viewers))
                        assert delivery.purge_settled and not delivery.may_hold_frames
                        if mode == "capture_purge":
                            release_capture.set()
                            # The result captured before suspension must be
                            # discarded, never sent after local purge settlement.
                            assert (await outgoing.get())["text"].startswith("purge:")
                        # No client acknowledgement can arrive after this
                        # disconnect, but no frame ever crossed the boundary.
                        await incoming.put({"type": "websocket.disconnect", "code": 1000})
                    elif mode == "purge_disconnect":
                        suspended = asyncio.create_task(
                            service.suspend_viewer_delivery(record.identity)
                        )
                        delivery = next(iter(service._viewers))
                        async with asyncio.timeout(1):
                            while delivery.purge_token is None:
                                await asyncio.sleep(0)
                        assert not suspended.done() and not delivery.purge_settled
                        await incoming.put({"type": "websocket.disconnect", "code": 1000})
                        await asyncio.wait_for(task, 5)
                        assert not suspended.done() and not delivery.purge_settled
                        suspended.cancel()
                        with pytest.raises(asyncio.CancelledError):
                            await suspended
                        assert suspended.cancelled() and suspended.cancelling() == 1
                    elif mode == "cancel_send":
                        await send_entered.wait()
                        task.cancel()
                        with pytest.raises(asyncio.CancelledError):
                            await task
                        assert task.cancelled() and task.cancelling() == 1
                    elif mode in {"takeover_purge", "takeover_during_capture", "retire"}:
                        if mode == "takeover_during_capture":
                            await asyncio.sleep(0.51)
                            await incoming.put({"type": "websocket.receive", "text": "frame"})
                            await capture_entered.wait()
                        if mode == "retire":
                            await incoming.put({"type": "websocket.receive", "text": "retire"})
                        else:
                            await owner.request_takeover(
                                principal=BrowserControlPrincipal(subject="operator"),
                                operator_session_id="continuity",
                                intent=intent_for(bootstrap),
                            )
                        if mode == "takeover_during_capture":
                            release_capture.set()
                        elif mode != "retire":
                            await incoming.put({"type": "websocket.receive", "text": "frame"})
                        purge = (await outgoing.get())["text"]
                        assert purge.startswith("purge:")
                        assert len(captures) == (2 if mode == "takeover_during_capture" else 1)
                        await incoming.put(
                            {"type": "websocket.receive", "text": "purged:" + purge[6:]}
                        )
                        await asyncio.wait_for(task, 2)
                        assert (await outgoing.get())["text"] == "retired"
                        # The sensitive-entry delivery barrier now has no stale
                        # closed viewer to wait on, without accepting a new frame.
                        await service.suspend_viewer_delivery(record.identity)
                    elif mode in {"purge", "wrong_purge"}:
                        suspended = asyncio.create_task(
                            service.suspend_viewer_delivery(record.identity)
                        )
                        purge = (await outgoing.get())["text"]
                        assert purge.startswith("purge:") and not suspended.done()
                        await incoming.put(
                            {
                                "type": "websocket.receive",
                                "text": "purged:"
                                + (
                                    "wrong"
                                    if mode == "wrong_purge"
                                    else purge.removeprefix("purge:")
                                ),
                            }
                        )
                        if mode == "wrong_purge":
                            await task
                            assert not suspended.done()
                            suspended.cancel()
                            with pytest.raises(asyncio.CancelledError):
                                await suspended
                        else:
                            await suspended
                    else:
                        await incoming.put({"type": "websocket.disconnect", "code": 1000})
            if mode != "cancel_send":
                await asyncio.wait_for(task, 5)
            assert len(captures) == (
                2
                if mode == "takeover_during_capture"
                else 1
                if mode
                in {
                    "valid",
                    "purge",
                    "cancel_send",
                    "wrong_purge",
                    "capture_purge",
                    "purge_disconnect",
                    "takeover_purge",
                    "retire",
                    "takeover_during_capture",
                }
                else 0
            )
            if mode in {"valid", "cancel_send", "wrong_purge", "purge_disconnect"}:
                assert len(service._viewers) == 1
                retained = next(iter(service._viewers))
                assert retained.closed and retained.may_hold_frames
                assert not retained.purge_settled
            else:
                assert not service._viewers
            if captures:
                assert captures[0]["principal"].subject == "operator"
            assert await owner.drain()

    asyncio.run(scenario())
