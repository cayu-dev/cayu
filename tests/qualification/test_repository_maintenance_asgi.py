"""Real mounted Runtime teardown waits for the admitted application request."""

import asyncio
import importlib

import pytest

from cayu import TaskQuery
from tests.qualification.test_repository_maintenance_application import project as project
from tests.qualification.test_repository_maintenance_http import client
from tests.qualification.test_repository_maintenance_http import host as host
from tests.qualification.test_repository_maintenance_request import consumer as consumer


@pytest.mark.parametrize(
    ("cancel_shutdown", "cancel_request", "invalid_message"),
    [
        (False, False, False),
        (False, True, False),
        (True, False, False),
        (True, True, False),
        (False, False, True),
    ],
)
def test_shutdown_seals_and_joins_before_runtime_drains(
    host, monkeypatch, cancel_shutdown, cancel_request, invalid_message
):
    server, application, registry, provider, _auth = host
    wrapper = importlib.import_module("operations.maintenance_asgi").MaintenanceASGI(server)

    async def scenario():
        incoming, outgoing = asyncio.Queue(), asyncio.Queue()
        entered, release, stopping = asyncio.Event(), asyncio.Event(), asyncio.Event()
        request_cancelled = asyncio.Event()
        events = []
        original_reserve = registry.reserve
        original_drain = application.app.drain_background_interruptions

        async def reserve(*args, **kwargs):
            events.append("request")
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                # Controlled adapter deliberately continues after server timeout.
                # Its later mutation must precede dependency teardown.
                request_cancelled.set()
                await release.wait()
            result = await original_reserve(*args, **kwargs)
            events.append("reserved")
            return result

        async def drain(**kwargs):
            events.append("runtime-drain")
            return await original_drain(**kwargs)

        async def receive():
            message = await incoming.get()
            if message["type"] != "lifespan.startup":
                stopping.set()
            return message

        monkeypatch.setattr(registry, "reserve", reserve)
        monkeypatch.setattr(application.app, "drain_background_interruptions", drain)
        lifetime = asyncio.create_task(
            wrapper({"type": "lifespan", "asgi": {"version": "3.0"}}, receive, outgoing.put)
        )
        request = None
        try:
            async with client(wrapper) as http:
                assert (await http.get("/runs/anything")).status_code == 503
                await incoming.put({"type": "lifespan.startup"})
                assert (await asyncio.wait_for(outgoing.get(), 5))[
                    "type"
                ] == "lifespan.startup.complete"
                request = asyncio.create_task(
                    http.post(
                        "/runs",
                        headers={"authorization": "Bearer tenant-a-token"},
                        json={"instruction": "Fix upper endpoint", "idempotency_key": "asgi-1"},
                    )
                )
                await asyncio.wait_for(entered.wait(), 5)
                if cancel_request:
                    request.cancel("server-timeout")
                    await asyncio.wait_for(request_cancelled.wait(), 5)
                    assert not request.done() and request.cancelling() == 1
                await incoming.put({"type": "unknown" if invalid_message else "lifespan.shutdown"})
                await asyncio.wait_for(stopping.wait(), 5)
                if cancel_shutdown:
                    lifetime.cancel("shutdown-observer")
                    await asyncio.sleep(0)
                assert not lifetime.done() and events == ["request"]
                assert (await http.get("/internal/cayu/api/sessions")).status_code == 503
                assert not request.done()
                release.set()
                response = await asyncio.wait_for(request, 5)
                assert response.status_code == 202
                assert not request.cancelled()
                assert request.cancelling() == (1 if cancel_request else 0)
                expected = (
                    "lifespan.shutdown.failed" if invalid_message else "lifespan.shutdown.complete"
                )
                assert (await asyncio.wait_for(outgoing.get(), 5))["type"] == expected
                if invalid_message:
                    with pytest.raises(ValueError, match="Invalid maintenance lifespan"):
                        await lifetime
                elif cancel_shutdown:
                    with pytest.raises(asyncio.CancelledError, match="shutdown-observer"):
                        await lifetime
                    assert lifetime.cancelled() and lifetime.cancelling() == 1
                else:
                    await lifetime
                assert events == ["request", "reserved", "runtime-drain"]
                assert (await http.get("/runs/anything")).status_code == 503
            tasks = await application.app.task_store.list_tasks(
                TaskQuery(type="maintenance.coding")
            )
            assert len(tasks) == 1 and not provider.requests
        finally:
            release.set()
            await incoming.put({"type": "lifespan.shutdown"})
            await asyncio.gather(lifetime, *([request] if request else []), return_exceptions=True)

    asyncio.run(scenario())


def test_failed_startup_never_opens_admission_or_restarts(host):
    module = importlib.import_module("operations.maintenance_asgi")

    async def scenario():
        calls, output = [], []

        async def failed(scope, receive, send):
            calls.append(scope["type"])
            await receive()
            await send({"type": "lifespan.startup.failed", "message": "unavailable"})

        async def receive():
            return {"type": "lifespan.startup"}

        async def send(message):
            output.append(message)

        wrapper = module.MaintenanceASGI(failed)
        await wrapper({"type": "websocket"}, receive, send)
        assert output[-1] == {"type": "websocket.close", "code": 1013}
        await wrapper({"type": "lifespan"}, receive, send)
        await wrapper({"type": "http"}, receive, send)
        assert output[-2]["status"] == 503
        with pytest.raises(RuntimeError, match="cannot restart"):
            await wrapper({"type": "lifespan"}, receive, send)
        assert calls == ["lifespan"]

    asyncio.run(scenario())
