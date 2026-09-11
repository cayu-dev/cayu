"""GitHub operator routes with real queues and controlled upstream evidence."""

import asyncio
import importlib
import json
import warnings

import pytest

from tests.qualification.test_repository_maintenance_git_http import OP, PRODUCT, server
from tests.qualification.test_repository_maintenance_github_intake import (
    github_intake as github_intake,
)
from tests.qualification.test_repository_maintenance_http import client
from tests.qualification.test_repository_maintenance_intake import intake as intake


def test_operator_reviews_and_queues_separate_github_consent(github_intake):
    _module, application, reservations, identity = github_intake
    api = server(application, reservations)
    path = f"/operator/runs/{identity.public_id}/github/approval?tenant=tenant-a"

    async def scenario():
        async with client(api) as http:
            response = await http.get(path, headers=OP)
            assert response.status_code == 200
            review = response.json()
            assert set(review) == {"id", "phase", "request_fingerprint", "request"}
            assert review["phase"] == "github_delivery" and review["request"]["merge"] is False
            body = {
                "request_fingerprint": review["request_fingerprint"],
                "approval_id": "separate-pr-consent",
            }
            wrong = dict(body, request_fingerprint="sha256:" + "0" * 64)
            assert (await http.post(path, headers=OP, json=wrong)).status_code == 409
            assert (
                await application.app.task_store.load_task(identity.github_delivery_task_id) is None
            )
            created = await http.post(path, headers=OP, json=body)
            assert created.status_code == 202
            assert created.json() == {
                "id": identity.public_id,
                "phase": "github_delivery",
                "task_status": "pending",
            }
            stored = await application.app.task_store.load_task(identity.github_delivery_task_id)
            assert stored.invocation.origin.subject == "maintenance-operator"
            assert json.loads(stored.input["request_json"]) == review["request"]
            assert json.loads(stored.input["approval_json"])["approval_id"] == "separate-pr-consent"
            assert (await http.post(path, headers=OP, json=body)).status_code == 202
            assert await application.app.task_store.load_task(stored.id) == stored

    asyncio.run(scenario())


@pytest.mark.parametrize("method", ["GET", "POST"])
def test_auth_and_tenant_lookup_precede_github_helpers(github_intake, monkeypatch, method):
    _module, application, reservations, identity = github_intake
    api = server(application, reservations)
    http_module = importlib.import_module("operations.maintenance_http")

    async def forbidden(*args, **kwargs):
        pytest.fail("Unauthorized lookup reached GitHub helper")

    monkeypatch.setattr(http_module, "load_github_approval_request", forbidden)
    monkeypatch.setattr(http_module, "ensure_github_delivery_task", forbidden)

    async def scenario():
        async with client(api) as http:
            for headers, query, status in (
                (PRODUCT, "tenant=tenant-a", 401),
                (OP, "tenant=other", 404),
                (OP, "tenant=tenant-a&tenant=other", 422),
            ):
                response = await http.request(
                    method,
                    f"/operator/runs/{identity.public_id}/github/approval?{query}",
                    headers=headers,
                    content=b"malformed",
                )
                assert response.status_code == status

    asyncio.run(scenario())


@pytest.mark.parametrize("bad", ["actor", "tree", "boolean", "unicode", "duplicate", "oversize"])
def test_malformed_github_approval_does_not_mutate_or_echo(
    github_intake, monkeypatch, caplog, capsys, bad
):
    _module, application, reservations, identity = github_intake
    api = server(application, reservations)
    http_module = importlib.import_module("operations.maintenance_http")
    canary = "private-pr-approval-canary"
    body: dict[str, object] = {"request_fingerprint": "sha256:" + "a" * 64, "approval_id": canary}
    if bad in {"actor", "tree"}:
        body["actor_subject" if bad == "actor" else "prepared_tree"] = canary
    elif bad == "boolean":
        body["request_fingerprint"] = True
    elif bad == "unicode":
        body["approval_id"] = "\ud800"
    raw = json.dumps(body).encode()
    if bad == "duplicate":
        raw = ('{"approval_id":"' + canary + '","approval_id":"again"}').encode()
    elif bad == "oversize":
        raw = (canary * 1000).encode()

    async def forbidden(*args, **kwargs):
        pytest.fail("Malformed request reached mutation")

    monkeypatch.setattr(http_module, "ensure_github_delivery_task", forbidden)

    async def scenario():
        async with client(api) as http:
            response = await http.post(
                f"/operator/runs/{identity.public_id}/github/approval?tenant=tenant-a",
                headers=OP,
                content=raw,
            )
            assert response.status_code == (413 if bad == "oversize" else 422)
            assert canary not in response.text
            assert (
                await application.app.task_store.load_task(identity.github_delivery_task_id) is None
            )

    with warnings.catch_warnings(record=True) as captured:
        asyncio.run(scenario())
    output = capsys.readouterr()
    assert not captured and canary not in caplog.text + output.out + output.err


def test_response_loss_replays_exact_github_task(github_intake, monkeypatch):
    _module, application, reservations, identity = github_intake
    api = server(application, reservations)
    http_module = importlib.import_module("operations.maintenance_http")
    original = http_module.ensure_github_delivery_task
    calls = 0

    async def lose_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = await original(*args, **kwargs)
        if calls == 1:
            raise ConnectionError("private-ack-canary")
        return result

    monkeypatch.setattr(http_module, "ensure_github_delivery_task", lose_once)

    async def scenario():
        path = f"/operator/runs/{identity.public_id}/github/approval?tenant=tenant-a"
        async with client(api) as http:
            review = (await http.get(path, headers=OP)).json()
            body = {
                "request_fingerprint": review["request_fingerprint"],
                "approval_id": "pr-consent",
            }
            failed = await http.post(path, headers=OP, json=body)
            assert failed.status_code == 503 and "private-ack-canary" not in failed.text
            stored = await application.app.task_store.load_task(identity.github_delivery_task_id)
            assert stored is not None
            assert (await http.post(path, headers=OP, json=body)).status_code == 202
            assert await application.app.task_store.load_task(stored.id) == stored

    asyncio.run(scenario())


def test_real_body_cancellation_does_not_queue(github_intake):
    _module, application, reservations, identity = github_intake
    api = server(application, reservations)

    async def scenario():
        entered = asyncio.Event()

        async def body():
            yield b'{"approval_id":'
            entered.set()
            await asyncio.Event().wait()

        async with client(api) as http:
            owner = asyncio.create_task(
                http.post(
                    f"/operator/runs/{identity.public_id}/github/approval?tenant=tenant-a",
                    headers=OP,
                    content=body(),
                )
            )
            try:
                await asyncio.wait_for(entered.wait(), 5)
                owner.cancel("stop-pr-body")
                with pytest.raises(asyncio.CancelledError, match="stop-pr-body"):
                    await owner
                assert owner.cancelled() and owner.cancelling() == 1
                assert (
                    await application.app.task_store.load_task(identity.github_delivery_task_id)
                    is None
                )
            finally:
                if not owner.done():
                    owner.cancel()
                await asyncio.gather(owner, return_exceptions=True)

    asyncio.run(scenario())
