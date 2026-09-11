"""Operator ASGI authorization and real queue writes; native reads controlled here."""

import asyncio
import importlib
import json
import warnings

import pytest

from cayu.server import ProductPrincipal
from tests.cli.test_scaffold_coding_budget import denial_policy
from tests.qualification.test_repository_maintenance_git_approval import (
    approval_context as approval_context,
)
from tests.qualification.test_repository_maintenance_git_intake import git_intake as git_intake
from tests.qualification.test_repository_maintenance_http import client
from tests.qualification.test_repository_maintenance_intake import intake as intake

OP = {"authorization": "Bearer operator-token"}
PRODUCT = {"authorization": "Bearer product-token"}


def server(application, reservations):
    application.app.budget_policy = denial_policy()
    auth = importlib.import_module("integrations.maintenance_auth")
    access = auth.MaintenanceAccess(
        product_tokens={"product-token": ProductPrincipal(tenant_id="tenant-a", subject_id="user")},
        operator_token="operator-token",
    )
    module = importlib.import_module("operations.maintenance_http")
    return module.build_maintenance_server(application, reservations, access)


def test_operator_preparation_uses_authenticated_origin_and_exact_queue(git_intake):
    _module, application, reservations, identity = git_intake
    api = server(application, reservations)
    path = f"/operator/runs/{identity.public_id}/git/preparation?tenant=tenant-a"

    async def scenario():
        async with client(api) as http:
            assert (await http.post(path, headers=PRODUCT, json={})).status_code == 401
            assert (
                await application.app.task_store.load_task(identity.git_preparation_task_id) is None
            )
            created = await http.post(path, headers=OP, json={})
            assert created.status_code == 202
            assert created.json() == {
                "id": identity.public_id,
                "phase": "git_preparation",
                "task_status": "pending",
            }
            replay = await http.post(path, headers=OP, json={})
            assert replay.json() == created.json()
        stored = await application.app.task_store.load_task(identity.git_preparation_task_id)
        assert stored.invocation.origin.subject == "maintenance-operator"
        assert stored.invocation.origin.tenant == identity.intent.tenant

    asyncio.run(scenario())


def test_operator_reads_and_approves_exact_pending_request(approval_context):
    _module, application, reservations, identity, native, _pending, _receipts, options = (
        approval_context
    )
    api = server(application, reservations)
    path = f"/operator/runs/{identity.public_id}/git/approval?tenant=tenant-a"

    async def scenario():
        async with client(api) as http:
            assert (await http.get(path, headers=PRODUCT)).status_code == 401
            response = await http.get(path, headers=OP)
            assert response.status_code == 200
            recorded = response.json()
            assert recorded["recorded_state"] == "approval_required"
            assert recorded["request"] == native.model_dump(mode="json")
            body = {
                "request_fingerprint": recorded["request_fingerprint"],
                "prepared_tree": recorded["prepared_tree"],
                "approval_id": options["approval_id"],
            }
            wrong = dict(body, prepared_tree="0" * 40)
            assert (await http.post(path, headers=OP, json=wrong)).status_code == 409
            assert await application.app.task_store.load_task(identity.git_delivery_task_id) is None
            assert (await http.post(path, headers=OP, json=body)).status_code == 202
            assert (await http.post(path, headers=OP, json=body)).status_code == 202
        stored = await application.app.task_store.load_task(identity.git_delivery_task_id)
        assert stored.invocation.origin.subject == "maintenance-operator"
        assert json.loads(stored.input["approval_json"])["approval_id"] == options["approval_id"]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "method,suffix", [("POST", "preparation"), ("GET", "approval"), ("POST", "approval")]
)
def test_operator_auth_and_tenant_precede_delivery_helpers(git_intake, monkeypatch, method, suffix):
    _module, application, reservations, identity = git_intake
    api = server(application, reservations)
    module = importlib.import_module("operations.maintenance_http")

    async def forbidden(*args, **kwargs):
        pytest.fail("Unauthorized request reached delivery helper")

    for name in (
        "ensure_git_preparation_task",
        "load_git_approval_request",
        "ensure_git_delivery_task",
    ):
        monkeypatch.setattr(module, name, forbidden)

    async def scenario():
        async with client(api) as http:
            prefix = f"/operator/runs/{identity.public_id}/git/{suffix}"
            for headers, query, status in (
                (PRODUCT, "tenant=tenant-a", 401),
                (OP, "tenant=other", 404),
                (OP, "tenant=tenant-a&tenant=other", 422),
            ):
                response = await http.request(
                    method, prefix + "?" + query, headers=headers, content=b"malformed"
                )
                assert response.status_code == status

    asyncio.run(scenario())


@pytest.mark.parametrize("bad", ["actor", "duplicate", "oversize", "bool", "unicode"])
def test_approval_body_is_bounded_and_not_echoed(
    approval_context, monkeypatch, bad, caplog, capsys
):
    _module, application, reservations, identity, native, _pending, _receipts, options = (
        approval_context
    )
    api = server(application, reservations)
    module = importlib.import_module("operations.maintenance_http")
    canary = "private-approval-canary"
    body = {
        "request_fingerprint": native.fingerprint,
        "prepared_tree": options["expected_tree"],
        "approval_id": canary,
    }
    if bad == "actor":
        body["actor_subject"] = canary
    elif bad == "bool":
        body["prepared_tree"] = True
    elif bad == "unicode":
        body["approval_id"] = "\ud800"
    raw = json.dumps(body).encode()
    if bad == "duplicate":
        raw = ('{"approval_id":"' + canary + '","approval_id":"again"}').encode()
    elif bad == "oversize":
        raw = (canary * 1000).encode()

    async def forbidden(*args, **kwargs):
        pytest.fail("Malformed approval reached mutation helper")

    monkeypatch.setattr(module, "ensure_git_delivery_task", forbidden)

    async def scenario():
        with warnings.catch_warnings(record=True) as recorded:
            async with client(api) as http:
                response = await http.post(
                    f"/operator/runs/{identity.public_id}/git/approval?tenant=tenant-a",
                    headers=OP,
                    content=raw,
                )
                assert response.status_code == (413 if bad == "oversize" else 422)
                assert canary not in response.text
            assert not recorded
        assert await application.app.task_store.load_task(identity.git_delivery_task_id) is None

    asyncio.run(scenario())
    output = capsys.readouterr()
    assert canary not in caplog.text + output.out + output.err


def test_http_approval_lost_response_keeps_one_exact_task(approval_context, monkeypatch):
    _module, application, reservations, identity, native, _pending, _receipts, options = (
        approval_context
    )
    api = server(application, reservations)
    module = importlib.import_module("operations.maintenance_http")
    original = module.ensure_git_delivery_task
    calls = 0

    async def lose_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        task = await original(*args, **kwargs)
        if calls == 1:
            raise ConnectionError("private-response-canary")
        return task

    monkeypatch.setattr(module, "ensure_git_delivery_task", lose_once)

    async def scenario():
        path = f"/operator/runs/{identity.public_id}/git/approval?tenant=tenant-a"
        body = {
            "request_fingerprint": native.fingerprint,
            "prepared_tree": options["expected_tree"],
            "approval_id": options["approval_id"],
        }
        async with client(api) as http:
            uncertain = await http.post(path, headers=OP, json=body)
            assert uncertain.status_code == 503
            assert "private-response-canary" not in uncertain.text
            committed = await application.app.task_store.load_task(identity.git_delivery_task_id)
            assert committed is not None
            replay = await http.post(path, headers=OP, json=body)
            assert replay.status_code == 202
            assert (
                await application.app.task_store.load_task(identity.git_delivery_task_id)
                == committed
            )

    asyncio.run(scenario())


def test_http_approval_body_cancellation_preserves_signal(approval_context):
    _module, application, reservations, identity, _native, _pending, _receipts, _options = (
        approval_context
    )
    api = server(application, reservations)

    async def scenario():
        entered = asyncio.Event()

        async def body():
            yield b'{"request_fingerprint":'
            entered.set()
            await asyncio.Event().wait()

        async with client(api) as http:
            request = asyncio.create_task(
                http.post(
                    f"/operator/runs/{identity.public_id}/git/approval?tenant=tenant-a",
                    headers=OP,
                    content=body(),
                )
            )
            await asyncio.wait_for(entered.wait(), timeout=5)
            request.cancel("stop-http-approval")
            with pytest.raises(asyncio.CancelledError, match="stop-http-approval"):
                await request
            assert request.cancelling() == 1 and request.cancelled()
            assert await application.app.task_store.load_task(identity.git_delivery_task_id) is None

    asyncio.run(scenario())
