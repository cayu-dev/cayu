"""Operator discovery reaches existing Runtime views, without recovery mutations."""

import asyncio
import importlib
import warnings

import pytest

from cayu import TaskQuery
from tests.qualification.test_repository_maintenance_http import (
    _A,
    _B,
    _BODY,
    _OP,
    client,
)
from tests.qualification.test_repository_maintenance_http import host as host
from tests.qualification.test_repository_maintenance_request import consumer as consumer
from tests.qualification.test_repository_maintenance_request import project as project


@pytest.mark.parametrize("consumer", ["memory", "sqlite"], indirect=True)
def test_operator_discovers_reserved_identity_and_runtime_view(host, monkeypatch):
    server, application, registry, provider, _auth = host

    async def scenario():
        async with client(server) as http:
            created = await http.post("/runs", headers=_A, json=_BODY)
            assert created.status_code == 202
            public_id = created.json()["id"]
            identity = await registry.load_owned(tenant="tenant-a", public_id=public_id)
            stores = importlib.import_module("operations.maintenance_runs")
            reopened = stores.SQLiteMaintenanceRunStore(registry.path)
            monkeypatch.setattr(registry, "load_owned", reopened.load_owned)
            url = f"/operator/runs/{public_id}?tenant=tenant-a"
            response = await http.get(url, headers=_OP)
            assert response.status_code == 200
            data = response.json()
            assert data == {
                **created.json(),
                "allocated_references": {
                    "product_run_id": identity.product_run_id,
                    "coding_session_id": application.app.project_session_id_for_exposure(
                        identity.session_id
                    ),
                    "workflow_session_id": application.app.project_session_id_for_exposure(
                        identity.workflow_session_id
                    ),
                    "tasks": {
                        "coding": identity.task_id,
                        "git_preparation": identity.git_preparation_task_id,
                        "git_delivery": identity.git_delivery_task_id,
                        "github_delivery": identity.github_delivery_task_id,
                    },
                },
            }
            task_view = await http.get(
                f"/internal/cayu/api/tasks/{data['allocated_references']['tasks']['coding']}",
                headers=_OP,
            )
            assert task_view.status_code == 200
            assert task_view.json()["id"] == identity.task_id
            assert task_view.json()["status"] == data["coding_task_status"]
            # Allocation is not evidence that coding has begun or completed.
            for field in ("coding_session_id", "workflow_session_id"):
                session = await http.get(
                    f"/internal/cayu/api/sessions/{data['allocated_references'][field]}",
                    headers=_OP,
                )
                assert session.status_code == 404
            product = await http.get(f"/runs/{public_id}", headers=_A)
            assert product.json() == created.json()
            for secret in (
                identity.intent.subject,
                identity.intent.request_json,
                str(application.project_root),
            ):
                assert secret not in response.text
            assert len(await application.app.task_store.list_tasks(TaskQuery())) == 1
            assert not provider.requests

    asyncio.run(scenario())


def test_operator_authentication_and_tenant_selection_precede_storage(host, monkeypatch):
    server, application, registry, _provider, _auth = host

    async def scenario():
        async with client(server) as http:
            created = await http.post("/runs", headers=_A, json=_BODY)
            public_id = created.json()["id"]
            url = f"/operator/runs/{public_id}"
            original = registry.load_owned

            async def forbidden(*args, **kwargs):
                pytest.fail("Rejected lookup reached storage")

            monkeypatch.setattr(registry, "load_owned", forbidden)
            for headers in ({}, _A, _B):
                assert (
                    await http.get(url + "?tenant=tenant-a", headers=headers)
                ).status_code == 401
            for query in ("", "?tenant=", "?tenant=a&tenant=b", "?tenant=%20a"):
                assert (await http.get(url + query, headers=_OP)).status_code == 422
            assert (
                await http.get("/operator/runs/not-a-uuid?tenant=tenant-a", headers=_OP)
            ).status_code == 404
            monkeypatch.setattr(registry, "load_owned", original)
            monkeypatch.setattr(application.app.task_store, "load_task", forbidden)
            assert (await http.get(url + "?tenant=tenant-b", headers=_OP)).status_code == 404

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["unavailable", "corrupt", "task-conflict", "projection"])
def test_operator_lookup_failure_never_publishes_partial_or_private_data(
    host, monkeypatch, caplog, capsys, failure
):
    server, application, registry, provider, _auth = host
    canary = "private-operator-canary"

    class Hostile:
        def __repr__(self):
            return canary

    async def scenario():
        async with client(server) as http:
            created = await http.post("/runs", headers=_A, json=_BODY)
            public_id = created.json()["id"]

            async def broken(**kwargs):
                if failure == "unavailable":
                    raise ConnectionError(canary)
                return Hostile()

            if failure == "task-conflict":
                identity = await registry.load_owned(tenant="tenant-a", public_id=public_id)
                task = await application.app.task_store.load_task(identity.task_id)

                async def wrong_task(_id):
                    return task.model_copy(update={"input": {"maintenance_run_id": canary}})

                monkeypatch.setattr(application.app.task_store, "load_task", wrong_task)
            elif failure == "projection":

                def unavailable_projection(_value):
                    raise ValueError(canary)

                monkeypatch.setattr(
                    application.app, "project_session_id_for_exposure", unavailable_projection
                )
            else:
                monkeypatch.setattr(registry, "load_owned", broken)
            response = await http.get(f"/operator/runs/{public_id}?tenant=tenant-a", headers=_OP)
            assert response.status_code == 503
            assert response.json() == {"detail": "Maintenance state unavailable."}
            assert not provider.requests

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        asyncio.run(scenario())
    assert all(canary not in str(item.message) for item in captured)
    output = capsys.readouterr()
    assert canary not in caplog.text + output.out + output.err


def test_operator_discovery_does_not_recreate_missing_task(host, monkeypatch):
    server, application, registry, provider, _auth = host

    async def scenario():
        reserve = registry.reserve
        committed = []

        async def lost_response(*args, **kwargs):
            committed.append(await reserve(*args, **kwargs))
            raise ConnectionError("response lost")

        monkeypatch.setattr(registry, "reserve", lost_response)
        async with client(server) as http:
            assert (await http.post("/runs", headers=_A, json=_BODY)).status_code == 503
            identity = committed[0]
            response = await http.get(
                f"/operator/runs/{identity.public_id}?tenant=tenant-a", headers=_OP
            )
            assert response.status_code == 200
            assert response.json()["coding_task_status"] == "intake_pending"
            assert response.json()["allocated_references"]["tasks"]["coding"] == identity.task_id
            assert await application.app.task_store.list_tasks(TaskQuery()) == []
            assert not provider.requests

    asyncio.run(scenario())


def test_operator_lookup_cancellation_preserves_signal_and_durable_state(host, monkeypatch):
    server, application, registry, provider, _auth = host

    async def scenario():
        async with client(server) as http:
            created = await http.post("/runs", headers=_A, json=_BODY)
            public_id = created.json()["id"]
            before = await application.app.task_store.list_tasks(TaskQuery())
            entered = asyncio.Event()

            async def blocked(**kwargs):
                entered.set()
                await asyncio.Event().wait()

            monkeypatch.setattr(registry, "load_owned", blocked)
            request = asyncio.create_task(
                http.get(f"/operator/runs/{public_id}?tenant=tenant-a", headers=_OP)
            )
            try:
                await asyncio.wait_for(entered.wait(), timeout=5)
                request.cancel("operator-disconnected")
                with pytest.raises(asyncio.CancelledError, match="operator-disconnected"):
                    await request
                assert request.cancelled() and request.cancelling() == 1
            finally:
                if not request.done():
                    request.cancel()
                await asyncio.gather(request, return_exceptions=True)
            assert await application.app.task_store.list_tasks(TaskQuery()) == before
            assert not provider.requests

    asyncio.run(scenario())
