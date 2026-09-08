"""Single-operation private input transport; no text in HTTP or durable receipts."""

import asyncio
from urllib.parse import urlsplit

from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketState

from cayu.runtime._browser_control_authorization import BrowserControlPermissionDenied
from cayu.runtime._browser_control_coordinator import BrowserControlCoordinator
from cayu.runtime._browser_control_input_tickets import BrowserInputTickets
from cayu.runtime._browser_control_service import BrowserControlService

OPERATOR_INPUT_SUBPROTOCOL = "cayu.browser-input.v1"


def create_browser_input_router(
    *,
    coordinator: BrowserControlCoordinator,
    service: BrowserControlService,
    tickets: BrowserInputTickets,
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
        raise ValueError("Browser input requires an exact HTTPS origin.")
    router = APIRouter(prefix="/browser-control")
    active: set[object] = set()

    @router.websocket("/input")
    async def input_once(socket: WebSocket):
        if (
            socket.url.scheme != "wss"
            or socket.query_params
            or socket.headers.getlist("origin") != [allowed_origin]
            or socket.scope.get("subprotocols") != [OPERATOR_INPUT_SUBPROTOCOL]
            or len(active) >= 32
        ):
            await socket.close(code=1008)
            return
        owner = object()
        active.add(owner)
        message = {}
        token = raw = text = None
        primary = cleanup = None
        try:
            async with asyncio.timeout(25):
                await socket.accept(subprotocol=OPERATOR_INPUT_SUBPROTOCOL)
                async with asyncio.timeout(5):
                    message = await socket.receive()
                token = message.get("text")
                message.clear()
                if type(token) is not str:
                    raise BrowserControlPermissionDenied()
                authenticated = tickets.consume(token)
                token = None
                # A ticket is not admission: recheck before asking for text,
                # and the serialized native owner repeats this before its CAS.
                await coordinator._prepare_text_input(
                    principal=authenticated.principal,
                    operator_session_id=authenticated.operator_session_id,
                    intent=authenticated.intent,
                )
                await socket.send_text("ready")
                async with asyncio.timeout(5):
                    message = await socket.receive()
                raw = message.get("bytes")
                message.clear()
                if type(raw) is not bytes or not 1 <= len(raw) <= 16384:
                    raise BrowserControlPermissionDenied()
                text = raw.decode("utf-8")
                raw = None
                if not 1 <= len(text) <= 4096 or "\x00" in text:
                    raise BrowserControlPermissionDenied()
                if (
                    authenticated.intent.input_kind != "text"
                    and text != authenticated.intent.input_kind
                ):
                    raise BrowserControlPermissionDenied()
                result = await service.submit_text_input(
                    principal=authenticated.principal,
                    operator_session_id=authenticated.operator_session_id,
                    intent=authenticated.intent,
                    text=text,
                )
                text = None
                await socket.send_json(
                    {
                        "state": "settled",
                        "revision": result.revision,
                        "control_epoch": result.control_epoch,
                        "settled_input_sequence": result.settled_input_sequence,
                    }
                )
        except Exception:
            # Transport/validation/extension failures never reflect credential
            # material. Real cancellation and process-control signals propagate.
            pass
        except BaseException as failure:
            primary = failure
        finally:
            token = raw = text = None
            message.clear()
            try:
                if socket.application_state is WebSocketState.CONNECTED:
                    async with asyncio.timeout(5):
                        await socket.close(code=1000)
            except BaseException as failure:
                cleanup = failure
            finally:
                active.discard(owner)
        if isinstance(primary, asyncio.CancelledError) and (
            cleanup is None or isinstance(cleanup, Exception)
        ):
            if cleanup is not None:
                raise primary from cleanup
            raise primary
        failures = [failure for failure in (primary, cleanup) if failure is not None]
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise BaseExceptionGroup("Browser input channel cleanup failed.", failures)

    return router
