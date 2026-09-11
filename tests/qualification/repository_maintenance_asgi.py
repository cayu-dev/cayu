"""Admission and request settlement around the maintained FastAPI lifespan."""

import asyncio


class MaintenanceASGI:
    """One process-local ASGI lifespan; call settlement is not remote-effect proof."""

    def __init__(self, app):
        self.app = app
        self._state = "new"
        self._active: set[asyncio.Future[None]] = set()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            await self._lifespan(scope, receive, send)
            return
        if scope["type"] not in {"http", "websocket"}:
            raise ValueError("Unsupported maintenance ASGI scope.")
        if self._state != "open":
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1013})
            else:
                await send({"type": "http.response.start", "status": 503, "headers": []})
                await send({"type": "http.response.body", "body": b"Service unavailable."})
            return
        completion = asyncio.get_running_loop().create_future()
        self._active.add(completion)
        try:
            await self.app(scope, receive, send)
        finally:
            completion.set_result(None)
            self._active.remove(completion)

    async def _lifespan(self, scope, receive, send):
        if self._state != "new":
            raise RuntimeError("Maintenance ASGI lifespan cannot restart.")
        self._state = "starting"
        cancellations = []

        async def seal_and_join():
            self._state = "closed"
            while self._active:
                try:
                    await asyncio.wait(tuple(self._active))
                except asyncio.CancelledError as exc:
                    cancellations.append(exc)

        async def owned_receive():
            try:
                message = await receive()
                expected = "lifespan.startup" if self._state == "starting" else "lifespan.shutdown"
                if type(message) is not dict or message.get("type") != expected:
                    raise ValueError("Invalid maintenance lifespan message.")
            except asyncio.CancelledError as exc:
                if self._state == "starting":
                    raise
                cancellations.append(exc)
                message = {"type": "lifespan.shutdown"}
            except BaseException:
                await seal_and_join()
                raise
            if message["type"] == "lifespan.shutdown":
                await seal_and_join()
            return message

        async def owned_send(message):
            await send(message)
            if message["type"] == "lifespan.startup.complete" and self._state == "starting":
                self._state = "open"

        try:
            await self.app(scope, owned_receive, owned_send)
        except BaseException as failure:
            if cancellations:
                raise cancellations[0] from failure
            raise
        finally:
            self._state = "closed"
        if cancellations:
            raise cancellations[0]
