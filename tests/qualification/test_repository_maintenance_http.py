"""Real ASGI product/operator boundaries; no network listener or provider call."""

import asyncio
import importlib
import warnings
from contextlib import asynccontextmanager
from decimal import Decimal

import httpx
import pytest

from cayu import BudgetWindow, TaskQuery
from cayu.server import ProductPrincipal
from tests.cli.test_scaffold_coding_budget import denial_policy
from tests.qualification.test_repository_maintenance_application import project as project
from tests.qualification.test_repository_maintenance_request import consumer as consumer

_A = {"authorization": "Bearer tenant-a-token"}
_B = {"authorization": "Bearer tenant-b-token"}
_OP = {"authorization": "Bearer operator-only-token"}
_BODY = {"instruction": "Repair the inclusive upper endpoint.", "idempotency_key": "fix-1"}


@pytest.fixture
def host(consumer, tmp_path):
    application, _task, provider, _domain, _requests, _workflow = consumer
    auth = importlib.import_module("integrations.maintenance_auth")
    api = importlib.import_module("operations.maintenance_http")
    stores = importlib.import_module("operations.maintenance_runs")
    registry = stores.SQLiteMaintenanceRunStore(tmp_path / "http-reservations.sqlite")
    asyncio.run(registry.initialize())
    access = auth.MaintenanceAccess(
        product_tokens={
            "tenant-a-token": ProductPrincipal(tenant_id="tenant-a", subject_id="alice"),
            "tenant-b-token": ProductPrincipal(tenant_id="tenant-b", subject_id="bob"),
        },
        operator_token="operator-only-token",
    )
    application.app.budget_policy = denial_policy()
    server = api.build_maintenance_server(application, registry, access)
    return server, application, registry, provider, auth


def client(server):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=server), base_url="http://fixture")


def lifecycle_server(host, lifespan):
    _server, application, registry, _provider, auth = host
    api = importlib.import_module("operations.maintenance_http")
    access = auth.MaintenanceAccess(
        product_tokens={
            "tenant-a-token": ProductPrincipal(tenant_id="tenant-a", subject_id="alice")
        },
        operator_token="operator-only-token",
    )
    return api.build_maintenance_server(application, registry, access, lifespan=lifespan)


@pytest.mark.parametrize("consumer", ["memory", "sqlite"], indirect=True)
@pytest.mark.parametrize("cancel", [False, True])
def test_host_lifespan_surrounds_actual_runtime_cleanup(host, monkeypatch, cancel):
    _server, application, _registry, provider, _auth = host
    events = []
    startup = ["recover_persisted_event_side_effects", "resume_pending_interruption_cascades"]
    drains = [
        "drain_background_interruptions",
        "drain_recovery_cleanups",
        "drain_provider_operation_cancellations",
        "drain_environment_cleanups",
        "drain_knowledge_publications",
    ]

    def observe(name, original):
        async def call(*args, **kwargs):
            result = await original(*args, **kwargs)
            if name in drains:
                assert result is True
            events.append(name)
            return result

        return call

    for name in startup + drains:
        monkeypatch.setattr(application.app, name, observe(name, getattr(application.app, name)))

    @asynccontextmanager
    async def owner(_server):
        events.append("host-enter")
        try:
            yield
        finally:
            # The fixture owns final store closure. This read proves those
            # dependencies are still usable at the outer owner's exit.
            await application.app.task_store.list_tasks(TaskQuery())
            events.append("host-exit")

    server = lifecycle_server(host, owner)

    async def scenario():
        entered = asyncio.Event()

        async def serve():
            async with server.router.lifespan_context(server):
                events.append("serving")
                async with client(server) as http:
                    accepted = await http.post("/runs", headers=_A, json=_BODY)
                    assert accepted.status_code == 202
                entered.set()
                if cancel:
                    await asyncio.Event().wait()

        task = asyncio.create_task(serve())
        try:
            await asyncio.wait_for(entered.wait(), timeout=10)
            if cancel:
                task.cancel("host-stop")
                with pytest.raises(asyncio.CancelledError, match="host-stop"):
                    await task
                assert task.cancelled() and task.cancelling() == 1
            else:
                await task
                assert not task.cancelled() and task.cancelling() == 0
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        assert events == ["host-enter", *startup, "serving", *drains, "host-exit"]
        assert not provider.requests

    asyncio.run(scenario())


def test_host_readiness_failure_precedes_runtime_startup(host, monkeypatch):
    _server, application, _registry, provider, _auth = host
    failure = RuntimeError("readiness failed")

    async def forbidden(*args, **kwargs):
        pytest.fail("Runtime startup ran before host readiness")

    monkeypatch.setattr(application.app, "recover_persisted_event_side_effects", forbidden)

    @asynccontextmanager
    async def owner(_server):
        raise failure
        yield  # pragma: no cover

    server = lifecycle_server(host, owner)

    async def scenario():
        with pytest.raises(RuntimeError) as caught:
            async with server.router.lifespan_context(server):
                pytest.fail("Host became available after failed readiness")
        assert caught.value is failure
        assert await application.app.task_store.list_tasks(TaskQuery()) == []
        assert not provider.requests

    asyncio.run(scenario())


def invalid_policy(case):
    policy = denial_policy()
    limit = policy.limits[0]
    if case == "missing":
        return None
    if case == "empty":
        policy.limits = ()
    elif case == "multiple":
        policy.limits = (limit, limit)
    elif case == "scope":
        limit.scope, limit.key = "agent", "coding"
    elif case == "window":
        limit.window = BudgetWindow.rolling(seconds=60)
    elif case == "currency":
        limit.currency = "EUR"
    elif case == "cap":
        limit.max_estimated_cost = Decimal("1.01")
    elif case == "reservation":
        limit.reservation = None
    elif case == "unpriced":
        limit.reservation, limit.allow_unpriced = None, True
    elif case == "action":
        limit.reservation, limit.action = None, "observe"
    elif case == "pricing":
        limit.pricing.prices = ()
    else:
        raise AssertionError("Unknown test case")
    return policy


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "empty",
        "multiple",
        "scope",
        "window",
        "currency",
        "cap",
        "reservation",
        "unpriced",
        "action",
        "pricing",
    ],
)
def test_host_and_intake_require_bounded_budget(host, case, monkeypatch):
    server, application, registry, provider, auth = host
    api = importlib.import_module("operations.maintenance_http")
    application.app.budget_policy = invalid_policy(case)
    access = auth.MaintenanceAccess(
        product_tokens={"product": ProductPrincipal(tenant_id="t", subject_id="s")},
        operator_token="operator",
    )
    with pytest.raises(ValueError, match="at most USD 1"):
        api.build_maintenance_server(application, registry, access)

    async def forbidden(*args, **kwargs):
        pytest.fail("Invalid budget reached reservation")

    monkeypatch.setattr(registry, "reserve", forbidden)

    async def scenario():
        async with client(server) as http:
            response = await http.post("/runs", headers=_A, json=_BODY)
            assert response.status_code == 503
            assert response.json() == {"detail": "Maintenance intake unavailable."}
        assert await application.app.task_store.list_tasks(TaskQuery()) == []
        assert not provider.requests

    asyncio.run(scenario())


def test_budget_copy_and_corrupt_field_diagnostics(host, caplog, capsys):
    _server, application, _registry, _provider, _auth = host
    budget = importlib.import_module("domain.maintenance_budget")
    original = denial_policy()
    original.limits[0].max_estimated_cost = Decimal("1")
    copied = budget.require_maintenance_budget(original)
    assert copied == original and copied is not original
    original.limits[0].max_estimated_cost = Decimal("2")
    assert copied.limits[0].max_estimated_cost == Decimal("1")
    canary = "private-budget-canary"

    class Hostile:
        def __repr__(self):
            return canary

        def __str__(self):
            return canary

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        for field in ("max_estimated_cost", "scope", "reservation", "pricing"):
            corrupt = denial_policy()
            corrupt.limits[0].pricing.price_book_version = canary
            setattr(corrupt.limits[0], field, Hostile())
            with pytest.raises(ValueError) as caught:
                budget.require_maintenance_budget(corrupt)
            assert canary not in str(caught.value) + repr(caught.value)
    assert all(canary not in str(item.message) for item in captured)
    output = capsys.readouterr()
    assert canary not in caplog.text + output.out + output.err
    assert application.app.budget_policy is not None


def test_budget_removal_blocks_new_intake_not_owned_readback(host):
    server, application, _registry, provider, _auth = host

    async def scenario():
        async with client(server) as http:
            accepted = await http.post("/runs", headers=_A, json=_BODY)
            assert accepted.status_code == 202
            application.app.budget_policy = None
            rejected = await http.post("/runs", headers=_A, json=_BODY)
            assert rejected.status_code == 503
            readback = await http.get(f"/runs/{accepted.json()['id']}", headers=_A)
            assert readback.status_code == 200 and readback.json() == accepted.json()
        assert len(await application.app.task_store.list_tasks(TaskQuery())) == 1
        assert not provider.requests

    asyncio.run(scenario())


@pytest.mark.parametrize("consumer", ["memory", "sqlite"], indirect=True)
def test_http_intake_replay_conflict_and_tenant_projection(host, monkeypatch):
    server, application, registry, provider, _auth = host

    async def scenario():
        async with client(server) as http:
            first = await http.post("/runs", headers=_A, json=_BODY)
            assert first.status_code == 202 and first.json()["coding_task_status"] == "pending"
            assert set(first.json()) == {"id", "coding_task_status"}
            public_id = first.json()["id"]
            identity = await registry.load_owned(tenant="tenant-a", public_id=public_id)
            assert identity is not None
            replay = await http.post("/runs", headers=_A, json=_BODY)
            assert replay.json() == first.json()
            assert await registry.load_owned(tenant="tenant-a", public_id=public_id) == identity
            conflict = await http.post("/runs", headers=_A, json={**_BODY, "instruction": "other"})
            assert conflict.status_code == 409
            own = await http.get(f"/runs/{public_id}", headers=_A)
            assert own.json() == first.json()

            async def forbidden_task(_id):
                raise AssertionError("Cross-tenant lookup reached Runtime.")

            monkeypatch.setattr(application.app.task_store, "load_task", forbidden_task)
            denied = await http.get(f"/runs/{public_id}?tenant=tenant-a", headers=_B)
            assert denied.status_code == 404
            assert identity.session_id not in denied.text + own.text
            assert not provider.requests
            assert await application.app.session_store.load(identity.session_id) is None

    asyncio.run(scenario())


def test_product_and_operator_credentials_are_separate(host, monkeypatch):
    server, _application, registry, provider, auth = host

    async def forbidden(*args, **kwargs):
        raise AssertionError("Unauthenticated request reached application storage.")

    monkeypatch.setattr(registry, "reserve", forbidden)

    async def scenario():
        async with client(server) as http:
            for headers in ({}, _OP, {"authorization": "Bearer wrong"}, {"x-cayu-dev-tenant": "a"}):
                response = await http.post("/runs", headers=headers, json=_BODY)
                assert response.status_code == 401
            for headers in ({}, _A, _B):
                response = await http.get("/internal/cayu/api/sessions", headers=headers)
                assert response.status_code == 401
            response = await http.get("/internal/cayu/api/sessions", headers=_OP)
            assert response.status_code == 200
        assert not provider.requests

    asyncio.run(scenario())
    with pytest.raises(ValueError, match="authentication configuration"):
        auth.MaintenanceAccess(
            product_tokens={"same": ProductPrincipal(tenant_id="t", subject_id="s")},
            operator_token="same",
        )


def test_invalid_bodies_are_bounded_nonmutating_and_sanitized(host, caplog, capsys):
    server, application, _registry, provider, _auth = host
    canary = "private-http-canary"

    async def scenario():
        async with client(server) as http:
            bodies = (
                b"[",
                b'"scalar"',
                b"\xff",
                b"{}",
                ('{"instruction":"' + canary + '","idempotency_key":false}').encode(),
                ('{"instruction":"' + canary + '","idempotency_key":"a","tenant":"x"}').encode(),
                b'{"instruction":"a","instruction":"b","idempotency_key":"x"}',
                b" " * 8193,
            )
            for body in bodies:
                response = await http.post("/runs", headers=_A, content=body)
                assert response.status_code in {413, 422}
                assert canary not in response.text
            tasks = await application.app.task_store.list_tasks(
                TaskQuery(type="maintenance.coding")
            )
            assert tasks == [] and not provider.requests

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        asyncio.run(scenario())
    assert all(canary not in str(item.message) for item in captured)
    output = capsys.readouterr()
    assert canary not in caplog.text + output.out + output.err


def test_committed_intake_response_loss_replays_same_identity(host, monkeypatch):
    server, application, registry, provider, _auth = host

    async def scenario():
        original = registry.reserve
        committed = []

        async def lost_response(*args, **kwargs):
            committed.append(await original(*args, **kwargs))
            raise ConnectionError("private storage error")

        monkeypatch.setattr(registry, "reserve", lost_response)
        async with client(server) as http:
            failed = await http.post("/runs", headers=_A, json=_BODY)
            assert failed.status_code == 503 and "private storage error" not in failed.text
            identity = committed[0]
            pending = await http.get(f"/runs/{identity.public_id}", headers=_A)
            assert pending.json()["coding_task_status"] == "intake_pending"
            assert await application.app.task_store.load_task(identity.task_id) is None
            monkeypatch.setattr(registry, "reserve", original)
            replay = await http.post("/runs", headers=_A, json=_BODY)
            assert replay.status_code == 202 and replay.json()["id"] == identity.public_id
            assert (
                await registry.load_owned(tenant="tenant-a", public_id=identity.public_id)
                == identity
            )
            assert not provider.requests

    asyncio.run(scenario())


def test_concurrent_http_intake_creates_one_task(host):
    server, application, _registry, provider, _auth = host

    async def scenario():
        async with client(server) as http:
            first, second = await asyncio.gather(
                http.post("/runs", headers=_A, json=_BODY),
                http.post("/runs", headers=_A, json=_BODY),
            )
            assert first.status_code == second.status_code == 202
            assert first.json() == second.json()
            tasks = await application.app.task_store.list_tasks(
                TaskQuery(type="maintenance.coding")
            )
            assert len(tasks) == 1 and not provider.requests

    asyncio.run(scenario())


def test_committed_task_response_loss_does_not_duplicate_or_reset(host, monkeypatch):
    server, application, _registry, provider, _auth = host

    async def scenario():
        original = application.app.create_task
        committed = []

        async def lost_response(request):
            committed.append(await original(request))
            raise ConnectionError("private task error")

        monkeypatch.setattr(application.app, "create_task", lost_response)
        async with client(server) as http:
            failed = await http.post("/runs", headers=_A, json=_BODY)
            assert failed.status_code == 503 and "private task error" not in failed.text
            replay = await http.post("/runs", headers=_A, json=_BODY)
            assert replay.status_code == 202 and len(committed) == 1
            assert replay.json()["id"] == committed[0].input["maintenance_run_id"]
            assert await application.app.task_store.load_task(committed[0].id) == committed[0]
            assert not provider.requests

    asyncio.run(scenario())


def test_http_body_cancellation_propagates_without_intake(host, monkeypatch):
    server, application, registry, provider, _auth = host

    async def forbidden(*args, **kwargs):
        raise AssertionError("Cancelled body reached reservation.")

    monkeypatch.setattr(registry, "reserve", forbidden)

    async def scenario():
        entered, stopped = asyncio.Event(), asyncio.Event()

        async def body():
            yield b'{"instruction":'
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        async with client(server) as http:
            owner = asyncio.create_task(http.post("/runs", headers=_A, content=body()))
            try:
                await asyncio.wait_for(entered.wait(), timeout=5)
                owner.cancel("client stopped")
                with pytest.raises(asyncio.CancelledError, match="client stopped"):
                    await owner
                assert owner.cancelled() and owner.cancelling() == 1 and stopped.is_set()
                tasks = await application.app.task_store.list_tasks(
                    TaskQuery(type="maintenance.coding")
                )
                assert tasks == [] and not provider.requests
            finally:
                if not owner.done():
                    owner.cancel()
                await asyncio.gather(owner, return_exceptions=True)

    asyncio.run(scenario())


def test_owned_read_rejects_corruption_and_hides_storage_errors(host, monkeypatch):
    server, application, registry, _provider, _auth = host

    async def scenario():
        async with client(server) as http:
            created = await http.post("/runs", headers=_A, json=_BODY)
            identity = await registry.load_owned(tenant="tenant-a", public_id=created.json()["id"])
            task = await application.app.task_store.load_task(identity.task_id)
            changed = task.model_copy(update={"input": {"maintenance_run_id": "private-canary"}})

            async def wrong_task(_id):
                return changed

            monkeypatch.setattr(application.app.task_store, "load_task", wrong_task)
            response = await http.get(f"/runs/{identity.public_id}", headers=_A)
            assert response.status_code == 503 and "private-canary" not in response.text

            async def unavailable(*args, **kwargs):
                raise ConnectionError("private-dsn-canary")

            monkeypatch.setattr(registry, "load_owned", unavailable)
            response = await http.get(f"/runs/{identity.public_id}", headers=_A)
            assert response.status_code == 503 and "private-dsn-canary" not in response.text

    asyncio.run(scenario())
