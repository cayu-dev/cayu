"""The webhook route of ``app.host_external_waits``.

FastAPI reads the endpoint's annotations at runtime, so this module keeps them
real classes rather than postponed strings.
"""

from typing import TYPE_CHECKING

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from cayu.sessions.external_waits import (
    ExternalWaitCapacityExceeded,
    ExternalWaitConflict,
    ExternalWaitUnavailable,
)

if TYPE_CHECKING:
    from cayu.external_wait_hosting import ExternalWaitHosting

# The wait owner's capacity frees as in-flight observations finish.
_CAPACITY_RETRY_AFTER_S = 5


def external_wait_delivery_router(hosting: "ExternalWaitHosting", path: str) -> APIRouter:
    from cayu.external_wait_hosting import ReceivedExternalEvent

    router = APIRouter()

    @router.post(path + "/{source}", include_in_schema=False)
    async def deliver_external_event(source: str, request: Request) -> JSONResponse:
        endpoint = hosting.sources.get(source)
        if endpoint is None:
            # The same answer as a refused request, so callers can't list sources.
            return JSONResponse({"code": "unauthenticated"}, status_code=401)
        body, refusal = await _bounded_body(request, hosting.max_request_bytes)
        if refusal is not None:
            return refusal
        replayed = False

        async def replay_receive():
            nonlocal replayed
            if replayed:
                return {"type": "http.disconnect"}
            replayed = True
            return {"type": "http.request", "body": body, "more_body": False}

        try:
            event = await endpoint.receive(Request(request.scope, receive=replay_receive))
        except (ValueError, KeyError, TypeError):
            return JSONResponse({"code": "invalid_event"}, status_code=422)
        if event is None:
            return JSONResponse({"code": "unauthenticated"}, status_code=401)
        if type(event) is not ReceivedExternalEvent:
            raise TypeError("An external event source must return a ReceivedExternalEvent.")
        headers = None
        try:
            status, content = await hosting._deliver_received(source, event)
        except PermissionError:
            status, content = 403, {"code": "forbidden"}
        except ExternalWaitConflict:
            status, content = 409, {"code": "conflict"}
        except ExternalWaitCapacityExceeded:
            status, content = 429, {"code": "capacity"}
            headers = {"Retry-After": str(_CAPACITY_RETRY_AFTER_S)}
        except ExternalWaitUnavailable:
            status, content = 503, {"code": "unavailable"}
        except ValueError:
            status, content = 422, {"code": "invalid_event"}
        return JSONResponse(content, status_code=status, headers=headers)

    hosting._delivery_endpoints.add(deliver_external_event)
    return router


async def _bounded_body(request: Request, limit: int) -> tuple[bytes, JSONResponse | None]:
    """Read at most ``limit`` bytes, refusing a larger body before anyone parses it."""
    declared = request.headers.getlist("content-length")
    if len(declared) > 1:
        return b"", JSONResponse({"code": "invalid_request"}, status_code=400)
    if declared:
        try:
            declared_bytes = int(declared[0])
        except ValueError:
            declared_bytes = -1
        if declared_bytes < 0:
            return b"", JSONResponse({"code": "invalid_request"}, status_code=400)
        if declared_bytes > limit:
            return b"", JSONResponse({"code": "too_large"}, status_code=413)
    received = bytearray()
    while True:
        message = await request.receive()
        if message["type"] != "http.request":
            return b"", JSONResponse({"code": "invalid_request"}, status_code=400)
        chunk = message.get("body", b"")
        if len(received) + len(chunk) > limit:
            return b"", JSONResponse({"code": "too_large"}, status_code=413)
        received.extend(chunk)
        if not message.get("more_body", False):
            return bytes(received), None
