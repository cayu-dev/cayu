"""Owned local HTTPS/WSS server for native browser acceptance."""

import asyncio
import socket
from contextlib import asynccontextmanager
from typing import Literal

import uvicorn


@asynccontextmanager
async def browser_control_tls_server(
    app, certificate_directory, *, lifespan: Literal["on", "off"] = "off"
):
    # The caller constructs the endpoint-dependent app before issuing requests.
    # Existing endpoint-dependent tests own app lifecycle themselves. Explicit
    # lifespan tests opt into the actual ASGI startup/shutdown path.
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(128)
        listener.setblocking(False)
        server = uvicorn.Server(
            uvicorn.Config(
                app,
                lifespan=lifespan,
                ssl_certfile=str(certificate_directory / "certificate.pem"),
                ssl_keyfile=str(certificate_directory / "key.pem"),
                ws="websockets-sansio",
                ws_per_message_deflate=False,
                log_config=None,
                access_log=False,
                timeout_graceful_shutdown=5,
            )
        )
        task = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            async with asyncio.timeout(5):
                while not server.started:
                    if task.done():
                        await task
                        raise AssertionError("Native acceptance server exited before startup.")
                    await asyncio.sleep(0.01)
            yield listener.getsockname()[1]
        finally:
            server.should_exit = True
            try:
                await asyncio.wait_for(asyncio.shield(task), 10)
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
