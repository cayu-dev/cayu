"""The webhook route of ``app.host_external_waits``: authenticated delivery, then a task."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from fastapi import FastAPI

from cayu import CayuApp
from cayu.external_wait_hosting import (
    EXTERNAL_WAIT_SERVICE_TASK_TYPE,
    ExternalEventSource,
    ReceivedExternalEvent,
)
from cayu.external_waits import ExternalWaitAccessPolicy, ExternalWaitContext
from cayu.sessions.external_waits import ExternalCorrelationRequest, ExternalWaitScope
from cayu.sessions.memory import InMemorySessionStore
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.queries import TaskQuery

WORKER = ExternalWaitContext(principal="worker")
SCOPE = ExternalWaitScope(application_scope="video", generation=1)
OTHER_SCOPE = ExternalWaitScope(application_scope="video", generation=2)


class Policy(ExternalWaitAccessPolicy):
    def authorize(self, context, *, scope, source, action):
        if context.principal == "renderer-webhook":
            return source == "renderer" and action == "deliver"
        return context.principal == "worker"


class Renderer(ExternalEventSource):
    source = "renderer"
    calls = 0

    async def receive(self, request):
        type(self).calls += 1
        signature = request.headers.get("x-signature")
        if signature not in {"good", "untrusted"}:
            return None
        body = await request.json()
        principal = "renderer-webhook" if signature == "good" else "untrusted"
        return ReceivedExternalEvent(
            context=ExternalWaitContext(principal=principal),
            scope=ExternalWaitScope(application_scope="video", generation=body["generation"]),
            correlation_key=body["job"],
            delivery_id=body["delivery"],
            payload_json=json.dumps(body["result"]),
        )


class Uploader(Renderer):
    source = "uploader"


def _hosting(scopes=None, *, stores=None):
    sessions, tasks = stores or (InMemorySessionStore(), InMemoryTaskStore())
    app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
    scopes = scopes or {SCOPE: ("renderer", "uploader")}
    served = {name for names in scopes.values() for name in names}
    hosting = app.host_external_waits(
        access_policy=Policy(),
        context=WORKER,
        scopes=scopes,
        sources=tuple(source() for source in (Renderer, Uploader) if source.source in served),
        delivery_path="/webhooks/waits",
    )
    server = FastAPI()
    server.include_router(hosting.delivery_router())
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=server), base_url="http://t")
    return app, hosting, client


async def _reserve(hosting, scope=SCOPE, source="renderer"):
    return await hosting.waits.reserve_correlation(
        ExternalCorrelationRequest(scope=scope, source=source, correlation_key="job-42"),
        context=WORKER,
    )


def _post(client, signature="good", *, source="renderer", job="job-42", result=None, generation=1):
    return client.post(
        f"/webhooks/waits/{source}",
        headers={} if signature is None else {"x-signature": signature},
        json={
            "generation": generation,
            "job": job,
            "delivery": "d-1",
            "result": result or {"url": "v.mp4"},
        },
    )


def test_an_authenticated_webhook_delivers_and_a_retry_converges():
    async def scenario():
        app, hosting, client = _hosting()
        await _reserve(hosting)
        first = await _post(client)
        assert first.status_code == 202
        assert first.json()["disposition"] == "accepted"
        retry = await _post(client)
        assert retry.status_code == 202 and retry.json() == first.json()
        # No session is parked on it yet, so nothing is scheduled.
        assert (
            await app.task_store.list_tasks(TaskQuery(type=EXTERNAL_WAIT_SERVICE_TASK_TYPE)) == []
        )
        changed = await _post(client, result={"url": "other.mp4"})
        assert changed.status_code == 409 and changed.json() == {"code": "conflict"}
        await client.aclose()
        await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("signature", "source", "job", "status", "code"),
    [
        (None, "renderer", "job-42", 401, "unauthenticated"),
        ("bad", "renderer", "job-42", 401, "unauthenticated"),
        ("untrusted", "renderer", "job-42", 403, "forbidden"),
        ("good", "uploader", "job-42", 403, "forbidden"),
        ("good", "renderer", "job-7", 404, "unknown_correlation"),
        # A refused caller learns nothing of which keys exist...
        ("untrusted", "renderer", "job-7", 403, "forbidden"),
        # ...or which sources do.
        ("good", "printer", "job-42", 401, "unauthenticated"),
    ],
)
def test_the_route_refuses_what_it_must_not_deliver(signature, source, job, status, code):
    async def scenario():
        app, hosting, client = _hosting()
        await _reserve(hosting)
        response = await _post(client, signature, source=source, job=job)
        assert response.status_code == status and response.json() == {"code": code}
        assert (
            await app.task_store.list_tasks(TaskQuery(type=EXTERNAL_WAIT_SERVICE_TASK_TYPE)) == []
        )
        await client.aclose()
        await app.aclose()

    asyncio.run(scenario())


def test_a_correlation_of_another_source_is_unknown_to_this_one():
    async def scenario():
        app, hosting, client = _hosting()
        await _reserve(hosting, source="uploader")
        response = await _post(client)
        assert response.status_code == 404 and response.json() == {"code": "unknown_correlation"}
        await client.aclose()
        await app.aclose()

    asyncio.run(scenario())


def test_a_replayed_webhook_cannot_reach_a_later_generations_wait():
    async def scenario():
        stores = (InMemorySessionStore(), InMemoryTaskStore())
        app, hosting, client = _hosting({SCOPE: ("renderer",)}, stores=stores)
        await _reserve(hosting)
        first = await _post(client)
        assert first.status_code == 202 and first.json()["disposition"] == "accepted"
        await client.aclose()
        await app.aclose()

        # The application moves to generation 2 and reuses the key.
        app, hosting, client = _hosting({OTHER_SCOPE: ("renderer",)}, stores=stores)
        later = await _reserve(hosting, OTHER_SCOPE)
        replay = await _post(client)
        assert replay.status_code == 404 and replay.json() == {"code": "unknown_correlation"}
        wait = await hosting.waits.inspect(later, context=WORKER)
        assert wait.outcome is None and wait.correlation == later
        await client.aclose()
        await app.aclose()

        # While both generations are served, the replay still lands on its own.
        app, hosting, client = _hosting(
            {SCOPE: ("renderer",), OTHER_SCOPE: ("renderer",)}, stores=stores
        )
        again = await _post(client)
        assert again.status_code == 202 and again.json() == first.json()
        wait = await hosting.waits.inspect(later, context=WORKER)
        assert wait.outcome is None
        assert (await _post(client, generation=3)).status_code == 404
        await client.aclose()
        await app.aclose()

    asyncio.run(scenario())


def test_an_oversized_payload_is_refused():
    async def scenario():
        app, hosting, client = _hosting()
        await _reserve(hosting)
        response = await _post(client, result={"blob": "x" * 70_000})
        assert response.status_code == 422 and response.json() == {"code": "invalid_event"}
        await client.aclose()
        await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("chunked", [False, True])
def test_an_oversized_body_is_refused_before_the_source_reads_it(chunked):
    async def scenario():
        app, hosting, client = _hosting()
        await _reserve(hosting)
        body = json.dumps(
            {"generation": 1, "job": "job-42", "delivery": "d-1", "blob": "x" * 90_000}
        ).encode()
        assert len(body) > hosting.max_request_bytes
        Renderer.calls = 0

        async def stream():
            for start in range(0, len(body), 8192):
                yield body[start : start + 8192]

        response = await client.post(
            "/webhooks/waits/renderer",
            headers={"x-signature": "good"},
            content=stream() if chunked else body,
        )
        assert response.status_code == 413 and response.json() == {"code": "too_large"}
        assert Renderer.calls == 0
        await client.aclose()
        await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "body",
    [
        b"{not json",
        b'{"generation": 1, "delivery": "d-1", "result": {}}',
        b'{"generation": 1, "job": "", "delivery": "d", "result": {}}',
    ],
    ids=["malformed-json", "missing-key", "invalid-identifier"],
)
def test_a_body_the_source_cannot_read_is_refused(body):
    async def scenario():
        app, hosting, client = _hosting()
        await _reserve(hosting)
        response = await client.post(
            "/webhooks/waits/renderer", headers={"x-signature": "good"}, content=body
        )
        assert response.status_code == 422 and response.json() == {"code": "invalid_event"}
        await client.aclose()
        await app.aclose()

    asyncio.run(scenario())


def test_capacity_answers_429_with_retry_after(monkeypatch):
    from cayu.sessions.external_waits import ExternalWaitCapacityExceeded

    async def scenario():
        app, hosting, client = _hosting()
        await _reserve(hosting)

        async def full(*args, **kwargs):
            raise ExternalWaitCapacityExceeded("full")

        monkeypatch.setattr(hosting, "deliver", full)
        response = await _post(client)
        assert response.status_code == 429 and response.json() == {"code": "capacity"}
        assert response.headers["retry-after"] == "5"
        await client.aclose()
        await app.aclose()

    asyncio.run(scenario())


def test_a_declared_oversized_body_is_refused_without_reading_it():
    async def scenario():
        app, hosting, _client = _hosting()
        server = FastAPI()
        server.include_router(hosting.delivery_router())
        sent = []

        async def receive():
            raise AssertionError("The body was read.")

        async def send(message):
            sent.append(message)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/webhooks/waits/renderer",
            "raw_path": b"/webhooks/waits/renderer",
            "query_string": b"",
            "root_path": "",
            "headers": [(b"content-length", str(hosting.max_request_bytes + 1).encode())],
            "client": ("127.0.0.1", 1),
            "server": ("t", 80),
        }
        await server(scope, receive, send)
        assert sent[0]["status"] == 413
        await _client.aclose()
        await app.aclose()

    asyncio.run(scenario())


def _service(app, tmp_path):
    from cayu import AgentSpec
    from cayu.server import (
        DevelopmentProductAccess,
        OpenAccess,
        ProductPrincipal,
        create_agent_service,
    )
    from cayu.storage.product_operations_sqlite import SQLiteProductOperationStore

    async def developer(request):
        return ProductPrincipal(tenant_id="dev", subject_id="dev")

    app.register_agent(AgentSpec(name="assistant", model="model"))
    return create_agent_service(
        app,
        agent_name="assistant",
        mode="development",
        product_access=DevelopmentProductAccess(dependency=developer),
        operator_access=OpenAccess(),
        product_store=SQLiteProductOperationStore(tmp_path / "product.sqlite"),
    )


def test_a_maintained_service_mounts_the_delivery_route(tmp_path):
    async def scenario():
        app, hosting, client = _hosting()
        await client.aclose()
        await _reserve(hosting)
        service = _service(app, tmp_path)
        assert hosting.serves_delivery(service.asgi_app)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=service.asgi_app), base_url="http://t"
        ) as client:
            assert (await _post(client, None)).status_code == 401
            response = await _post(client)
            assert response.status_code == 202 and response.json()["disposition"] == "accepted"
        await app.aclose()

    asyncio.run(scenario())


def test_a_delivery_path_inside_the_services_own_paths_is_refused(tmp_path):
    app = CayuApp(
        session_store=InMemorySessionStore(), task_store=InMemoryTaskStore(), enable_logging=False
    )
    app.host_external_waits(
        access_policy=Policy(),
        context=WORKER,
        scopes={SCOPE: ("renderer",)},
        sources=(Renderer(),),
        delivery_path="/api/waits",
    )
    with pytest.raises(ValueError, match="must be separate from the service's /api routes"):
        _service(app, tmp_path)


def test_a_refused_caller_cannot_tell_which_generations_are_served():
    async def scenario():
        app, hosting, client = _hosting()
        await _reserve(hosting)
        response = await _post(client, "untrusted", generation=9)
        assert response.status_code == 403 and response.json() == {"code": "forbidden"}
        await client.aclose()
        await app.aclose()

    asyncio.run(scenario())
