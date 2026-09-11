"""Whole ASGI factory/lifespan with controlled native dependency boundaries."""

import asyncio
import importlib
import json
from types import SimpleNamespace

import pytest

from tests.qualification.test_repository_maintenance_application import project as project
from tests.qualification.test_repository_maintenance_http import client
from tests.qualification.test_repository_maintenance_http import host as host
from tests.qualification.test_repository_maintenance_request import consumer as consumer


@pytest.fixture
def api_host(host, monkeypatch):
    _server, application, reservations, _provider, _auth = host
    module = importlib.import_module("operations.maintenance_api")
    monkeypatch.setenv(
        "CAYU_MAINTENANCE_ACCESS_JSON",
        json.dumps(
            {
                "operator_token": "operator-only-token",
                "product_tokens": {
                    "tenant-a-token": {"tenant_id": "tenant-a", "subject_id": "alice"}
                },
            }
        ),
    )
    calls = []

    def build():
        calls.append("build")
        return application.app

    async def ready():
        calls.append("ready")

    async def close(**kwargs):
        calls.append("close")
        return True

    owned = SimpleNamespace(
        application=application,
        reservations=reservations,
        validate_startup_schema=ready,
        aclose=close,
    )

    def bind(app, *, agent_name):
        assert app is application.app and agent_name == application.agent_name
        calls.append("bind")
        return owned

    monkeypatch.setattr(module, "build_maintenance_app", build)
    monkeypatch.setattr(module, "bind_maintenance_deployment", bind)
    return module, owned, calls


def test_api_credentials_are_required_before_application_construction(api_host, monkeypatch):
    module, _owned, calls = api_host
    monkeypatch.delenv("CAYU_MAINTENANCE_ACCESS_JSON")
    with pytest.raises(ValueError, match="maintenance access configuration"):
        module.build_api()
    assert not calls


@pytest.mark.parametrize("cancel", [False, True])
def test_api_factory_owns_readiness_requests_and_cancellation_safe_close(
    api_host, monkeypatch, cancel
):
    module, owned, calls = api_host

    async def scenario():
        incoming, outgoing = asyncio.Queue(), asyncio.Queue()
        closing, release = asyncio.Event(), asyncio.Event()
        original = owned.application.app.drain_background_interruptions

        async def drain(**kwargs):
            calls.append("runtime-drain")
            return await original(**kwargs)

        async def close(**kwargs):
            calls.append("close-start")
            closing.set()
            await release.wait()
            calls.append("close-end")
            return True

        monkeypatch.setattr(owned.application.app, "drain_background_interruptions", drain)
        monkeypatch.setattr(owned, "aclose", close)
        server = module.build_api()
        assert calls == ["build", "bind"]
        task = asyncio.create_task(server({"type": "lifespan"}, incoming.get, outgoing.put))
        try:
            async with client(server) as http:
                assert (await http.get("/runs/unused")).status_code == 503
                await incoming.put({"type": "lifespan.startup"})
                assert (await asyncio.wait_for(outgoing.get(), 5))[
                    "type"
                ] == "lifespan.startup.complete"
                response = await http.get(
                    "/internal/cayu/api/sessions",
                    headers={"authorization": "Bearer operator-only-token"},
                )
                assert response.status_code == 200
                await incoming.put({"type": "lifespan.shutdown"})
                await asyncio.wait_for(closing.wait(), 5)
                assert calls == ["build", "bind", "ready", "runtime-drain", "close-start"]
                if cancel:
                    task.cancel("api-shutdown-observer")
                    await asyncio.sleep(0)
                assert not task.done()
                assert (await http.get("/runs/unused")).status_code == 503
                release.set()
                if cancel:
                    with pytest.raises(asyncio.CancelledError, match="api-shutdown-observer"):
                        await task
                    assert task.cancelled() and task.cancelling() == 1
                else:
                    await task
                assert calls[-1] == "close-end"
        finally:
            release.set()
            await incoming.put({"type": "lifespan.shutdown"})
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("cleanup_error", [False, True])
def test_startup_failure_still_closes_and_retains_original_errors(
    api_host, monkeypatch, cleanup_error
):
    module, owned, calls = api_host
    primary, cleanup = ValueError("startup failure"), OSError("cleanup failure")

    async def ready():
        raise primary

    async def close(**kwargs):
        calls.append("close")
        if cleanup_error:
            raise cleanup
        return True

    monkeypatch.setattr(owned, "validate_startup_schema", ready)
    monkeypatch.setattr(owned, "aclose", close)

    async def scenario():
        server = module.build_api()
        output = []

        async def receive():
            return {"type": "lifespan.startup"}

        async def send(message):
            output.append(message)

        with pytest.raises(ExceptionGroup if cleanup_error else ValueError) as caught:
            await server({"type": "lifespan"}, receive, send)
        if cleanup_error:
            assert isinstance(caught.value, ExceptionGroup)
            assert caught.value.exceptions == (primary, cleanup)
        else:
            assert caught.value is primary
        assert calls == ["build", "bind", "close"]
        assert output[0]["type"] == "lifespan.startup.failed"
        async with client(server) as http:
            assert (await http.get("/runs/unused")).status_code == 503

    asyncio.run(scenario())
