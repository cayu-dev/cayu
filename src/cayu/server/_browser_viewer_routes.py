"""Protected pull-based operator frames; no model/tool/event transport."""

import asyncio
import time
from urllib.parse import urlsplit

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from cayu.runtime._browser_control_authorization import (
    BrowserControlPermissionDenied,
    BrowserControlRevisionChanged,
)
from cayu.runtime._browser_control_coordinator import BrowserControlCoordinator
from cayu.runtime._browser_control_frames import BrowserViewUnavailable
from cayu.runtime._browser_control_service import BrowserControlService
from cayu.runtime._browser_control_view_tickets import BrowserViewTickets
from cayu.runtime.browser_control import BrowserControlConflict

OPERATOR_VIEW_SUBPROTOCOL = "cayu.browser-view.v1"


def create_browser_viewer_router(
    *,
    coordinator: BrowserControlCoordinator,
    service: BrowserControlService,
    tickets: BrowserViewTickets,
    allowed_origin: str,
) -> APIRouter:
    origin = urlsplit(allowed_origin)
    if (
        origin.scheme != "https"
        or not origin.netloc
        or origin.username is not None
        or origin.path
        or origin.query
        or origin.fragment
    ):
        raise ValueError("Browser viewer requires an exact HTTPS origin.")
    router = APIRouter(prefix="/browser-control")
    active: set[object] = set()

    @router.websocket("/viewer")
    async def viewer(socket: WebSocket):
        if (
            socket.url.scheme != "wss"
            or socket.query_params
            or socket.headers.getlist("origin") != [allowed_origin]
            or socket.scope.get("subprotocols") != [OPERATOR_VIEW_SUBPROTOCOL]
            or len(active) >= 32
        ):
            await socket.close(code=1008)
            return
        owner = object()
        active.add(owner)
        delivery = None
        try:
            await socket.accept(subprotocol=OPERATOR_VIEW_SUBPROTOCOL)
            async with asyncio.timeout(5):
                message = await socket.receive()
            token = message.get("text")
            message.clear()
            try:
                if type(token) is not str:
                    raise BrowserControlPermissionDenied()
                authenticated = tickets.consume(token)
            finally:
                token = None
            # Consuming a ticket authenticates HTTP continuity only. Current
            # application permission and durable generation still govern viewing.
            await coordinator.authorize_view(
                principal=authenticated.principal,
                operator_session_id=authenticated.operator_session_id,
                identity=authenticated.intent.identity,
                expected_record_revision=authenticated.intent.expected_record_revision,
            )
            delivery = service.register_viewer(authenticated.intent.identity)
            await socket.send_text("ready")
            last_capture = float("-inf")
            async with asyncio.timeout(600):
                for _ in range(1200):
                    idle_until = time.monotonic() + 30
                    while True:
                        if delivery.purge_token is not None:
                            token = delivery.purge_token
                            async with asyncio.timeout(5):
                                await socket.send_text("purge:" + token)
                                # At most one frame request and one intentional
                                # retirement may already be in flight. Drain those
                                # controls without dispatching or extending time.
                                for _ in range(3):
                                    acknowledgement = await socket.receive()
                                    if acknowledgement.get("text") not in {"frame", "retire"}:
                                        break
                            if acknowledgement.get("text") != "purged:" + token:
                                raise BrowserControlPermissionDenied()
                            delivery.acknowledge_purge(token)
                            async with asyncio.timeout(5):
                                await socket.send_text("retired")
                            return
                        try:
                            async with asyncio.timeout(0.25):
                                message = await socket.receive()
                        except TimeoutError:
                            if time.monotonic() >= idle_until:
                                raise
                            continue
                        if message["type"] == "websocket.disconnect":
                            return
                        if delivery.purge_token is None:
                            break
                    if message.get("text") == "retire":
                        delivery.request_purge()
                        continue
                    _, current = await coordinator._load(authenticated.intent.identity)
                    if current.revision != authenticated.intent.expected_record_revision:
                        # This ticket cannot admit another capture, but closing
                        # now would strand ownership of already delivered pixels.
                        # Keep only the bounded purge/ack exchange above alive.
                        delivery.request_purge()
                        continue
                    if message.get("text") != "frame" or time.monotonic() - last_capture < 0.5:
                        raise BrowserControlPermissionDenied()
                    last_capture = time.monotonic()
                    try:
                        frame = await service.capture_view(
                            identity=authenticated.intent.identity,
                            principal=authenticated.principal,
                            operator_session_id=authenticated.operator_session_id,
                            page=authenticated.intent.page,
                            until_ms=coordinator._view_grant_until(),
                        )
                    except (
                        BrowserViewUnavailable,
                        BrowserControlPermissionDenied,
                        BrowserControlRevisionChanged,
                    ):
                        delivery.request_purge()
                        continue
                    try:
                        if delivery.purge_token is not None:
                            continue
                        delivery.begin_send()
                        async with asyncio.timeout(5):
                            await socket.send_bytes(frame.png)
                    finally:
                        delivery.finish_send()
                        del frame
        except (
            BrowserControlPermissionDenied,
            BrowserControlConflict,
            TimeoutError,
            WebSocketDisconnect,
        ):
            pass
        finally:
            try:
                if socket.application_state is WebSocketState.CONNECTED:
                    async with asyncio.timeout(5):
                        await socket.close(code=1000)
            finally:
                if delivery is not None:
                    service.retire_viewer(delivery)
                active.discard(owner)

    return router
