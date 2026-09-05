"""Acceptance-only launcher for delivering a crash to one exact Chromium page.

The acceptance runner copies this script into its disposable allocation. It is
not part of the browser worker protocol or the model-facing tool surface.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import runpy
import sys

UPLOAD_FAULT = None


async def main(session_id: str) -> None:
    worker = runpy.run_path("/opt/cayu-browser/worker.py", run_name="cayu_acceptance_worker")
    daemon_type = worker["_InteractiveDaemon"]
    original_start = daemon_type.start
    fault_server: asyncio.AbstractServer | None = None
    socket_path = worker["_interactive_socket_path"](session_id).with_suffix(".page-fault.sock")
    if UPLOAD_FAULT is not None:
        if UPLOAD_FAULT not in {"disconnection", "acknowledgement_loss"}:
            raise ValueError("Unknown acceptance upload fault.")
        original_upload = daemon_type._upload_files

        async def upload_with_fault(daemon, state, request, locator, internal_ref):
            class SelectionBoundary:
                async def evaluate(self, expression):
                    return await locator.evaluate(expression)

                async def set_input_files(self, files):
                    # Deliver through the real browser selection and page effect,
                    # then lose acknowledgement before the worker publishes success.
                    async with state.page.expect_response(
                        lambda response: response.url.endswith("/effect/upload-selected"),
                        timeout=5000,
                    ) as selected:
                        await locator.set_input_files(files)
                    await selected.value
                    if UPLOAD_FAULT == "disconnection":
                        await daemon.browser.close()
                    raise RuntimeError("Acceptance upload acknowledgement was lost.")

            return await original_upload(daemon, state, request, SelectionBoundary(), internal_ref)

        daemon_type._upload_files = upload_with_fault

    async def start(daemon):
        nonlocal fault_server
        await original_start(daemon)

        async def handle(reader, writer):
            try:
                async with asyncio.timeout(5):
                    raw = json.loads(await reader.readuntil(b"\n"))
                    if type(raw) is not dict or set(raw) != {"page_id"}:
                        raise ValueError("Invalid page crash request.")
                    async with daemon.lock:
                        state = daemon.pages[raw["page_id"]]
                        if state.lifecycle not in {"active", "background"} or state.cdp is None:
                            raise ValueError("Page is not an exact live target.")
                        crashed = asyncio.Event()
                        state.page.once("crash", lambda *_: crashed.set())
                        crash_task = asyncio.create_task(state.cdp.send("Page.crash"))
                        try:
                            await crashed.wait()
                        finally:
                            # Page.crash normally rejects when its target dies.
                            crash_task.cancel()
                            with contextlib.suppress(Exception, asyncio.CancelledError):
                                await crash_task
                        if state.lifecycle != "crashed" or daemon.closing:
                            raise RuntimeError("Exact page crash was not observed.")
                        response = {"page_id": state.page_id, "crashed": True}
            except Exception:
                response = {"crashed": False}
            writer.write(json.dumps(response).encode() + b"\n")
            with contextlib.suppress(ConnectionError):
                await writer.drain()
            writer.close()
            with contextlib.suppress(ConnectionError):
                await writer.wait_closed()

        fault_server = await asyncio.start_unix_server(handle, path=str(socket_path), limit=1024)
        os.chmod(socket_path, 0o600)

    daemon_type.start = start
    try:
        await worker["_interactive_daemon_main"](session_id)
    finally:
        if fault_server is not None:
            fault_server.close()
            await fault_server.wait_closed()
        with contextlib.suppress(FileNotFoundError):
            socket_path.unlink()


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--interactive-daemon":
        raise SystemExit(2)
    asyncio.run(main(sys.argv[2]))
