"""Private allocation-authenticated guest channel, separate from operator access."""

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from cayu._task_wait import consume_pending_task_cancellation, restore_task_cancellation_requests
from cayu.runtime._browser_control_authorization import BrowserControlPermissionDenied
from cayu.runtime._browser_control_channel import (
    BrowserGuestCommandOwner,
    bind_browser_guest_channel,
)
from cayu.runtime._browser_control_channels import BrowserGuestChannelSettlement
from cayu.runtime._browser_control_coordinator import (
    BrowserControlCoordinator,
    BrowserControlInvocationEnded,
)
from cayu.runtime._browser_control_frames import MAX_BROWSER_FRAME_MESSAGE_BYTES
from cayu.runtime._browser_control_service import BrowserControlService
from cayu.runtime.browser_control import BrowserControlConflict
from cayu.tools._browser_control_transport import CONTROL_MESSAGE_BYTES, CONTROL_SUBPROTOCOL


class _GuestSocket:
    def __init__(self, socket: WebSocket) -> None:
        self.socket = socket

    async def recv(self):
        message = await self.socket.receive()
        if message["type"] == "websocket.disconnect":
            raise WebSocketDisconnect(message.get("code", 1000))
        value = message.get("text")
        if type(value) is not str or len(value.encode("utf-8")) > CONTROL_MESSAGE_BYTES:
            raise BrowserControlConflict("Browser guest control message is invalid.")
        return value

    async def send(self, value: str) -> None:
        await self.socket.send_text(value)

    async def recv_frame(self) -> bytes:
        message = await self.socket.receive()
        if message["type"] == "websocket.disconnect":
            raise WebSocketDisconnect(message.get("code", 1000))
        value = message.get("bytes")
        if type(value) is not bytes or not 4 < len(value) <= MAX_BROWSER_FRAME_MESSAGE_BYTES:
            raise BrowserControlConflict("Browser private frame message is invalid.")
        return value

    async def close(self) -> None:
        if self.socket.application_state is WebSocketState.CONNECTED:
            await self.socket.close(code=1000)

    async def idle(self) -> bool:
        try:
            async with asyncio.timeout(0.25):
                await self.recv()
        except TimeoutError:
            return True
        except WebSocketDisconnect:
            return False
        raise BrowserControlConflict("Browser guest sent unsolicited control data.")


def create_browser_guest_router(
    *, service: BrowserControlService, coordinator: BrowserControlCoordinator
) -> APIRouter:
    router = APIRouter(prefix="/browser-control")

    @router.websocket("/guest")
    async def guest(socket: WebSocket):
        await service.channels.run(lambda settlement: serve_guest(socket, settlement))

    async def serve_guest(socket: WebSocket, settlement: BrowserGuestChannelSettlement):
        try:
            if (
                socket.url.scheme != "wss"
                or socket.query_params
                or socket.headers.get("origin") is not None
                or socket.scope.get("subprotocols") != [CONTROL_SUBPROTOCOL]
            ):
                raise BrowserControlPermissionDenied()
            headers = socket.headers.getlist("authorization")
            if len(headers) != 1 or not headers[0].startswith("Bearer "):
                raise BrowserControlPermissionDenied()
            allocation = service.authenticate_guest(headers[0][7:])
            headers.clear()
        except BrowserControlPermissionDenied:
            await socket.close(code=1008)
            return
        await socket.accept(subprotocol=CONTROL_SUBPROTOCOL)
        connection = _GuestSocket(socket)
        bound = None
        command_owner = None
        primary = None
        cleanup = []
        try:
            bound = await bind_browser_guest_channel(
                coordinator=coordinator, allocation=allocation, connection=connection
            )
            service.confirm_guest(bound)
            command_owner = BrowserGuestCommandOwner(
                coordinator=coordinator,
                connection=connection,
                bound=bound,
                suspend_viewer_delivery=service.suspend_viewer_delivery,
            )
            service.attach_commands(command_owner)
            async with asyncio.timeout(3600):
                while True:
                    if await command_owner.step() is False:
                        break
                    if not await connection.idle():
                        break
        except (WebSocketDisconnect, BrowserControlInvocationEnded):
            pass
        except BaseException as error:
            primary = error
        finally:
            consumed = 0
            task = asyncio.current_task()
            if isinstance(primary, asyncio.CancelledError) and task is not None:
                before = task.cancelling()
                consume_pending_task_cancellation(primary)
                consumed = before - task.cancelling()
            if command_owner is not None or bound is not None:
                try:
                    if command_owner is not None:
                        await command_owner.disconnect()
                    elif bound is not None:
                        await coordinator._fence_idle_channel(expected=bound.record)
                except BaseException as error:
                    cleanup.append(error)
            try:
                await connection.close()
            except BaseException as error:
                cleanup.append(error)
            if command_owner is not None and not cleanup:
                try:
                    service.detach_commands(command_owner)
                except BaseException as error:
                    cleanup.append(error)
                else:
                    settlement.confirm()
            if consumed and isinstance(primary, asyncio.CancelledError):
                restore_task_cancellation_requests(consumed, cancellation=primary)
        if isinstance(primary, asyncio.CancelledError):
            if cleanup:
                raise primary from BaseExceptionGroup("Browser guest cleanup failed.", cleanup)
            raise primary
        failures = ([] if primary is None else [primary]) + cleanup
        if failures:
            raise BaseExceptionGroup("Browser guest channel failed.", failures)

    return router
