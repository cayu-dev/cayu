"""The explicit close response owns daemon shutdown until its channel drains."""

import asyncio
import json

import pytest
from tests.core.test_browser_session import _interactive_raw_request

from cayu.tools import _browser_guest as guest


@pytest.mark.parametrize("competing", [None, "list_pages", "close"])
@pytest.mark.parametrize("cleanup_ok", [True, False])
def test_explicit_close_drains_response_before_daemon_shutdown(
    monkeypatch, tmp_path, cleanup_ok, competing, *, response_end="drain"
):
    async def scenario():
        draining = asyncio.Event()
        release = asyncio.Event()
        retired = asyncio.Event()
        handler_tasks = []
        payloads = []
        handler_callback = None
        socket_path = tmp_path / "session.sock"

        class Daemon(guest._InteractiveDaemon):
            async def start(self):
                pass

            async def _ensure_configuration(self, request):
                pass

            async def _refresh_egress_trust(self):
                pass

            async def close(self, **kwargs):
                self.closing = True
                return cleanup_ok

        class Reader:
            def __init__(self, operation="close"):
                self.operation = operation

            async def readuntil(self, separator):
                return json.dumps(_interactive_raw_request(self.operation)).encode() + separator

        class Writer:
            def __init__(self, *, gated=True):
                self.gated = gated

            def write(self, payload):
                payloads.append(payload)

            async def drain(self):
                if self.gated:
                    draining.set()
                    await release.wait()

            def close(self):
                pass

            async def wait_closed(self):
                pass

        class Server:
            def close(self):
                retired.set()

            async def wait_closed(self):
                pass

        async def start_server(handler, **kwargs):
            nonlocal handler_callback
            handler_callback = handler
            socket_path.touch()
            handler_tasks.append(asyncio.create_task(handler(Reader(), Writer())))
            return Server()

        monkeypatch.setattr(guest, "_InteractiveDaemon", Daemon)
        monkeypatch.setattr(guest, "_interactive_socket_path", lambda _: socket_path)
        monkeypatch.setattr(guest, "_record_interactive_retirement", lambda _: True)
        monkeypatch.setattr(guest, "_INTERACTIVE_IDLE_POLL_SECONDS", 0.001)
        if response_end == "timeout":
            monkeypatch.setattr(guest, "_INTERACTIVE_RESPONSE_DRAIN_SECONDS", 0.02)
        monkeypatch.setattr(asyncio, "start_unix_server", start_server)
        main = asyncio.create_task(guest._interactive_daemon_main("bs_test"))
        try:
            async with asyncio.timeout(2):
                await draining.wait()
            if competing is not None:
                assert handler_callback is not None
                # A rejected request or exact receipt replay can drain first;
                # neither owns the original explicit-close response channel.
                peer = asyncio.create_task(handler_callback(Reader(competing), Writer(gated=False)))
                handler_tasks.append(peer)
                await peer
            if response_end == "drain":
                # The watcher runs while output is blocked after native close.
                with pytest.raises(TimeoutError):
                    async with asyncio.timeout(0.05):
                        await retired.wait()
                assert not main.done()
                release.set()
            elif response_end == "cancel":
                handler_tasks[0].cancel()
            async with asyncio.timeout(2):
                results = await asyncio.gather(main, *handler_tasks, return_exceptions=True)
            assert results[0] == 0
            if response_end == "cancel":
                assert isinstance(results[1], asyncio.CancelledError)
            else:
                assert all(result is None for result in results[1:])
            assert retired.is_set()
            result = json.loads(payloads[0])
            if cleanup_ok:
                assert result["kind"] == "closed"
                assert result["allocation_disposition"] == "retired"
            else:
                assert result["error"] == "cleanup_failed"
        finally:
            release.set()
            if not main.done():
                main.cancel()
            await asyncio.gather(main, *handler_tasks, return_exceptions=True)

    asyncio.run(scenario())


def test_lost_explicit_close_acknowledgement_remains_ambiguous_and_is_not_replayed(tmp_path):
    from tests.core.test_browser_session import _context, _FakeBrowserBackend

    from cayu.tools.browser_session import BrowserSessionTool

    async def scenario():
        backend = _FakeBrowserBackend()
        tool = BrowserSessionTool(_backend=backend)
        context = _context(tmp_path)
        opened = await tool.run(
            context,
            {
                "operation": "navigate",
                "url": "https://example.test",
                "operation_id": "navigate",
            },
        )
        assert not opened.is_error
        backend.failure = ConnectionError("lost close response")
        request = {
            "operation": "close",
            "operation_id": "close",
            "session_id": opened.structured["session_id"],
        }
        first = await tool.run(context, request)
        assert first.is_error
        assert first.structured["error"] == "outcome_ambiguous"
        assert first.structured["execution"]["dispatch"] == "acknowledgement_lost"
        assert first.structured.get("closed") is not True
        backend.failure = None
        assert await tool.run(context, request) == first
        assert [call["operation"] for call in backend.calls] == ["navigate", "close"]

    asyncio.run(scenario())


@pytest.mark.parametrize("response_end", ["timeout", "cancel"])
def test_failed_close_response_does_not_strand_daemon_shutdown(monkeypatch, tmp_path, response_end):
    test_explicit_close_drains_response_before_daemon_shutdown(
        monkeypatch, tmp_path, True, None, response_end=response_end
    )
