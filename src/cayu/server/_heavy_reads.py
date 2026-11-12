"""Keep heavy dashboard reads from delaying light ones, and stop them when nobody waits.

Dashboard aggregates (overview counts, session lists with overviews, usage rollups,
tool usage) scan many sessions or events. Three parts keep them in their place:

- :class:`HeavyReadLimiter` admits a bounded number of heavy reads at once, after the
  route's own authentication. Admitted reads run on the bulk store lane
  (:func:`cayu.storage._read_lanes.bulk_reads`), so they lease the store's bulk
  connections and leave the interactive ones to single-session reads.
- :class:`ClientDisconnectMiddleware` owns each request's ASGI receive channel and
  cancels a heavy read whose client has disconnected before the response started.
  Cancelling a SQLite read interrupts its statement; a Postgres read is cancelled
  through its driver.
- The limiter's FastAPI dependency ties the two together on a route.

Each part works alone: without the middleware a heavy read is still bounded and
laned, it just finishes after a disconnect.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from fastapi import Depends, Request

from cayu.server._request_timing import CLIENT_DISCONNECTED_SCOPE_KEY
from cayu.storage._read_lanes import bulk_reads

if TYPE_CHECKING:
    from starlette.types import ASGIApp, Message, Receive, Scope, Send

DEFAULT_HEAVY_READ_CONCURRENCY = 2
"""Heavy reads one limiter admits at once; the SQLite bulk lane has as many readers."""

_SCOPE_KEY = "cayu.client_disconnect"


class _RequestChannel:
    """One HTTP request's receive channel, shared by the app and a disconnect watch.

    Only one side reads the server's ``receive`` at a time. A watch reads it only
    after the app has received the whole request body, or at once when the request
    declares no body, holding that one empty request message for the app. While the
    watch reads, the app's receive waits for the disconnect the watch sees.
    """

    def __init__(self, scope: Scope, receive: Receive, task: asyncio.Task[Any]) -> None:
        self._receive = receive
        self._task = task
        self._declares_body = _declares_body(scope)
        self._held: deque[Message] = deque()
        self._app_reading = False
        self._app_read_done = asyncio.Event()
        self._watch_reading = False
        self._body_received = asyncio.Event()
        self._disconnected = asyncio.Event()
        self.response_started = False
        self.cancelled_for_disconnect = False

    async def receive(self) -> Message:
        if self._held:
            return self._held.popleft()
        if self._watch_reading or self._disconnected.is_set():
            await self._disconnected.wait()
            return {"type": "http.disconnect"}
        self._app_reading = True
        self._app_read_done.clear()
        try:
            message = await self._receive()
        finally:
            self._app_reading = False
            self._app_read_done.set()
        self._note(message)
        return message

    def _note(self, message: Message) -> None:
        if message["type"] == "http.disconnect":
            self._disconnected.set()
            self._body_received.set()
        elif message["type"] == "http.request" and not message.get("more_body", False):
            self._body_received.set()

    async def wait_for_disconnect(self) -> None:
        """Return once the client disconnects; never returns for a client that stays."""

        while not self._disconnected.is_set():
            if self._app_reading:
                await self._app_read_done.wait()
                continue
            if not self._body_received.is_set() and (self._declares_body or self._held):
                # The app reads the body itself, under the route's own size limits.
                await self._body_received.wait()
                continue
            self._watch_reading = True
            try:
                message = await self._receive()
            finally:
                self._watch_reading = False
            if message["type"] == "http.request" and not self._body_received.is_set():
                # The bodyless request's own message: hold it for the app.
                self._held.append(message)
            self._note(message)

    def watch(self) -> Callable[[], object]:
        """Cancel the request if its client disconnects before the response starts.

        Returns a function that ends the watch; call it when the heavy read ends.
        """

        watcher = asyncio.create_task(self._cancel_on_disconnect(), name="cayu-disconnect-watch")
        return watcher.cancel

    async def _cancel_on_disconnect(self) -> None:
        await self.wait_for_disconnect()
        if not self.response_started and not self.cancelled_for_disconnect:
            self.cancelled_for_disconnect = True
            self._task.cancel()


def _declares_body(scope: Scope) -> bool:
    for name, value in scope.get("headers", ()):
        if name == b"transfer-encoding":
            return True
        if name == b"content-length" and value.strip() not in {b"", b"0"}:
            return True
    return False


class ClientDisconnectMiddleware:
    """Let heavy reads stop when their client disconnects.

    Starlette runs a JSON endpoint to completion even after the client has gone. This
    middleware gives the request a shared receive channel that a heavy read's
    dependency can watch; a disconnect then cancels the request and the middleware
    ends it quietly, without a response, since nobody is left to read one.
    ``path_prefix`` limits it to the control-plane API when mounted into a host app.
    """

    def __init__(self, app: ASGIApp, *, path_prefix: str = "") -> None:
        self.app = app
        self._path_prefix = path_prefix.rstrip("/")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not _under(scope, self._path_prefix):
            await self.app(scope, receive, send)
            return
        task = asyncio.current_task()
        if task is None:
            await self.app(scope, receive, send)
            return
        channel = _RequestChannel(scope, receive, task)
        scope[_SCOPE_KEY] = channel

        async def observed_send(message: Message) -> None:
            if message["type"] == "http.response.start":
                channel.response_started = True
            await send(message)

        try:
            await self.app(scope, channel.receive, observed_send)
        except asyncio.CancelledError:
            if channel.cancelled_for_disconnect and task.uncancel() == 0:
                scope[CLIENT_DISCONNECTED_SCOPE_KEY] = True
                return
            raise
        else:
            if channel.cancelled_for_disconnect:
                # The app absorbed the cancellation; balance the request we made.
                task.uncancel()


def _under(scope: Scope, prefix: str) -> bool:
    if not prefix:
        return True
    path: str = scope.get("path", "")
    root_path: str = scope.get("root_path", "")
    candidates = [path]
    if root_path and path.startswith(root_path):
        candidates.append(path[len(root_path) :])
    return any(item == prefix or item.startswith(prefix + "/") for item in candidates)


class HeavyReadLimiter:
    """Admit a bounded number of heavy reads at once and run them on the bulk lane.

    Use :meth:`dependency` on a route, after its authentication dependency, so
    unauthenticated requests never wait for or take a slot.
    """

    def __init__(self, limit: int = DEFAULT_HEAVY_READ_CONCURRENCY) -> None:
        if type(limit) is not int or limit < 1:
            raise ValueError("Heavy read limit must be a positive integer.")
        self.limit = limit
        self._slots = asyncio.Semaphore(limit)
        self._running = 0
        self._waiting = 0

    @property
    def running(self) -> int:
        """Heavy reads admitted and still running."""

        return self._running

    @property
    def waiting(self) -> int:
        """Heavy reads waiting for a slot."""

        return self._waiting

    @asynccontextmanager
    async def _hold(self, request: Request) -> AsyncIterator[None]:
        channel = request.scope.get(_SCOPE_KEY)
        stop_watch = channel.watch() if isinstance(channel, _RequestChannel) else None
        try:
            self._waiting += 1
            try:
                await self._slots.acquire()
            finally:
                self._waiting -= 1
            self._running += 1
            try:
                with bulk_reads():
                    yield
            finally:
                self._running -= 1
                self._slots.release()
        finally:
            if stop_watch is not None:
                stop_watch()

    def dependency(self, applies: Callable[[Request], bool] | None = None) -> Any:
        """A FastAPI dependency that holds a slot while the endpoint runs.

        ``applies`` picks the requests that are heavy, for a route that serves both
        kinds; the others run at once, on the interactive lane.
        """

        async def admitted(request: Request) -> AsyncIterator[None]:
            if applies is not None and not applies(request):
                yield
                return
            async with self._hold(request):
                yield

        return Depends(admitted)


__all__ = [
    "DEFAULT_HEAVY_READ_CONCURRENCY",
    "ClientDisconnectMiddleware",
    "HeavyReadLimiter",
]
