"""Private guest bootstrap owns connection startup without public receipts."""

import asyncio
import sys
import tempfile
from pathlib import Path

import pytest

from cayu.tools import _browser_guest
from cayu.tools._browser_control_guest import GuestControlFailure


@pytest.mark.parametrize("kind", ["loss", "chained", "grouped", "ordinary", "cleanup"])
def test_allocation_loss_is_retired_only_after_successful_native_cleanup(kind):
    from cayu.tools._browser_control_guest import GuestControlAllocationLost

    async def scenario():
        daemon = _browser_guest._InteractiveDaemon("bs_loss")
        daemon.control.bind("a" * 64)
        release = asyncio.Event()
        entered = asyncio.Event()
        cleanup_failure = RuntimeError("native cleanup failed")
        loss = GuestControlAllocationLost()
        if kind == "chained":
            loss.__cause__ = cleanup_failure
        failure = (
            ExceptionGroup("channel and cleanup", [loss, cleanup_failure])
            if kind == "grouped"
            else GuestControlFailure()
            if kind == "ordinary"
            else loss
        )

        class Context:
            async def close(self):
                entered.set()
                await release.wait()
                if kind == "cleanup":
                    raise cleanup_failure

        async def stopped_channel():
            return (failure,)

        daemon.context = Context()
        daemon._operator_bootstrap_task = asyncio.create_task(stopped_channel())
        try:
            assert not await daemon.close(timeout_seconds=0.05)
            assert entered.is_set()
            owner = daemon.session_cleanup_tasks["context"]
            assert not owner.done()
            assert daemon.control.state == "control_uncertain"
            release.set()
            assert await daemon.close() is (kind == "loss")
            assert owner.done()
            assert daemon._operator_bootstrap_task.result() == (failure,)
            if kind == "cleanup":
                assert owner.exception() is cleanup_failure
                assert daemon.context is not None
            else:
                assert daemon.context is None
            assert daemon.control.state == ("closed" if kind == "loss" else "control_uncertain")
        finally:
            release.set()
            await daemon.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "response",
    [
        None,
        {"schema_version": True, "bootstrap_accepted": True},
        {"schema_version": 1, "bootstrap_accepted": 1},
    ],
)
def test_bootstrap_requires_exact_acceptance_without_replay(monkeypatch, response):
    async def scenario():
        calls = []

        async def send(path, raw):
            calls.append(1)
            return response

        monkeypatch.setattr(_browser_guest, "_interactive_send", send)
        with pytest.raises(_browser_guest._GuestFailure):
            await _browser_guest._run_private_control_bootstrap(
                {
                    "protocol_version": _browser_guest.CONTROL_BOOTSTRAP_PROTOCOL,
                    "session_id": "bs_ipc",
                    "endpoint": "wss://control.example/guest",
                    "credential": "c" * 64,
                    "scope_sha256": "d" * 64,
                }
            )
        assert calls == [1]

    asyncio.run(scenario())


def test_bootstrap_retains_connect_owner_and_never_records_bearer(monkeypatch):
    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()
        calls = []

        async def connect(*, endpoint, credential):
            calls.append(endpoint)
            assert credential == "a" * 64
            entered.set()
            await release.wait()
            raise OSError("connection unavailable")

        monkeypatch.setattr(_browser_guest, "open_guest_control_channel", connect)
        daemon = _browser_guest._InteractiveDaemon("bs_bootstrap")
        raw = {
            "endpoint": "wss://control.example/guest",
            "credential": "a" * 64,
            "scope_sha256": "b" * 64,
        }
        result = await daemon.bootstrap_operator_channel(raw)
        assert result == {"schema_version": 1, "bootstrap_accepted": True}
        await entered.wait()
        assert daemon._operator_bootstrap_task is not None
        assert not daemon._operator_bootstrap_task.done()
        with pytest.raises(GuestControlFailure):
            await daemon.bootstrap_operator_channel(raw)
        assert not daemon.operations
        assert not daemon.profile_operations
        release.set()
        errors = await daemon._operator_bootstrap_task
        assert len(errors) == 1 and isinstance(errors[0], OSError)
        # Failed startup remains owned, not automatically retried with a spent capability.
        with pytest.raises(GuestControlFailure):
            await daemon.bootstrap_operator_channel(raw)
        assert calls == ["wss://control.example/guest"]

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel", [False, True])
def test_shutdown_owns_connection_that_arrives_after_close_started(monkeypatch, cancel):
    async def scenario():
        connecting = asyncio.Event()
        connected = asyncio.Event()
        closing = asyncio.Event()
        closed = asyncio.Event()
        calls = []

        class Connection:
            async def close(self):
                calls.append("close")
                closing.set()
                await closed.wait()

        async def connect(**kwargs):
            connecting.set()
            await connected.wait()
            return Connection()

        monkeypatch.setattr(_browser_guest, "open_guest_control_channel", connect)
        daemon = _browser_guest._InteractiveDaemon("bs_shutdown")
        await daemon.bootstrap_operator_channel(
            {
                "endpoint": "wss://control.example/guest",
                "credential": "a" * 64,
                "scope_sha256": "b" * 64,
            }
        )
        await connecting.wait()
        first_close = asyncio.create_task(daemon.close(timeout_seconds=0.05))
        while not daemon.closing:
            await asyncio.sleep(0)
        connected.set()
        await closing.wait()
        if cancel:
            assert first_close.cancel("shutdown caller left")
            with pytest.raises(asyncio.CancelledError):
                await first_close
            assert first_close.cancelled()
            assert first_close.cancelling() == 1
        else:
            assert not await first_close
        assert daemon._operator_bootstrap_task is not None
        assert not daemon._operator_bootstrap_task.done()
        assert not daemon.session_cleanup_tasks["operator-transport"].done()
        with pytest.raises(GuestControlFailure):
            await daemon.bootstrap_operator_channel(
                {
                    "endpoint": "wss://control.example/guest",
                    "credential": "a" * 64,
                    "scope_sha256": "b" * 64,
                }
            )
        closed.set()
        assert await daemon._operator_bootstrap_task == ()
        assert await daemon.close()
        assert daemon._operator_connection is None
        assert "operator-transport" not in daemon.session_cleanup_tasks
        assert calls == ["close"]

    asyncio.run(scenario())


@pytest.mark.skipif(sys.platform == "win32", reason="The browser guest uses POSIX Unix sockets.")
def test_private_bootstrap_traverses_real_daemon_socket(monkeypatch):
    async def scenario(socket_path):
        observed = []
        dispatched = asyncio.Event()

        async def start(daemon):
            observed.append(daemon)

        async def close(daemon):
            if daemon._operator_bootstrap_task is not None:
                await daemon._operator_bootstrap_task
            return True

        async def connect(*, endpoint, credential):
            assert endpoint == "wss://control.example/guest"
            assert credential == "c" * 64
            dispatched.set()
            raise OSError("connection unavailable")

        monkeypatch.setattr(_browser_guest._InteractiveDaemon, "start", start)
        monkeypatch.setattr(_browser_guest._InteractiveDaemon, "close", close)
        monkeypatch.setattr(_browser_guest, "_interactive_socket_path", lambda _: socket_path)
        monkeypatch.setattr(_browser_guest, "_record_interactive_retirement", lambda _: None)
        monkeypatch.setattr(_browser_guest, "open_guest_control_channel", connect)
        server = asyncio.create_task(_browser_guest._interactive_daemon_main("bs_ipc"))
        try:
            async with asyncio.timeout(5):
                while not socket_path.exists():
                    if server.done():
                        await server
                    await asyncio.sleep(0.001)
            response = await _browser_guest._run_private_control_bootstrap(
                {
                    "protocol_version": _browser_guest.CONTROL_BOOTSTRAP_PROTOCOL,
                    "session_id": "bs_ipc",
                    "endpoint": "wss://control.example/guest",
                    "credential": "c" * 64,
                    "scope_sha256": "d" * 64,
                }
            )
            assert response == {"schema_version": 1, "bootstrap_accepted": True}
            await dispatched.wait()
            assert not observed[0].operations
            assert not observed[0].profile_operations
        finally:
            if observed:
                observed[0].close_requested.set()
            await server

    with tempfile.TemporaryDirectory(prefix="cayu-bc-", dir="/tmp") as directory:
        asyncio.run(scenario(Path(directory) / "guest.sock"))
