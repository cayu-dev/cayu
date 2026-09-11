"""Generated HTTP projection; full native delivery is exercised by the journey fixture."""

import asyncio
import importlib
import json
import warnings
from dataclasses import replace
from types import SimpleNamespace

import pytest

from cayu import TaskQuery, github_connector_behavior_fingerprint
from tests.qualification.test_repository_maintenance_delivery_configuration import authority
from tests.qualification.test_repository_maintenance_github_intake import configuration
from tests.qualification.test_repository_maintenance_http import _A, _B, _BODY, _OP, client
from tests.qualification.test_repository_maintenance_http import host as host
from tests.qualification.test_repository_maintenance_request import consumer as consumer
from tests.qualification.test_repository_maintenance_request import project as project


@pytest.fixture
def final_context(host, monkeypatch):
    server, application, registry, provider, _auth = host
    module = importlib.import_module("operations.maintenance_final")
    github = configuration()
    github["security"]["connector_behavior_fingerprint"] = github_connector_behavior_fingerprint()
    monkeypatch.setenv("CAYU_MAINTENANCE_GIT_JSON", json.dumps(authority()))
    monkeypatch.setenv("CAYU_MAINTENANCE_GITHUB_JSON", json.dumps(github))
    monkeypatch.setenv(
        "CAYU_MAINTENANCE_GITHUB_HOST_JSON",
        json.dumps(
            {
                "owner": "fixture",
                "repository_name": "maintenance",
                "api_base_url": "https://api.github.example",
            }
        ),
    )
    monkeypatch.setenv("CAYU_MAINTENANCE_GITHUB_WEB_ORIGIN", "https://github.example")
    digest = "a" * 64
    fingerprint = "sha256:" + digest
    accepted = module.MaintenanceAcceptance(
        "reference", digest, fingerprint, fingerprint, fingerprint, fingerprint
    )
    product = SimpleNamespace(
        result_reference=SimpleNamespace(reference_id="reference", digest=digest),
        candidate=SimpleNamespace(request_fingerprint=fingerprint, final_revision=fingerprint),
    )
    remote = SimpleNamespace(
        result=SimpleNamespace(tree="a" * 40), artifact=SimpleNamespace(sha256=fingerprint)
    )
    result = SimpleNamespace(
        repository_id="target",
        installation_id="installation",
        account_id="account",
        head_commit="b" * 40,
        pull_request=SimpleNamespace(
            number=7, url="https://github.example/fixture/maintenance/pull/7"
        ),
    )
    publication = SimpleNamespace(result=result, artifact=SimpleNamespace(sha256=fingerprint))
    reads = []

    async def verified(_application, _reservations, identity):
        assert _application is application and _reservations is registry
        reads.append(identity.public_id)
        return None, product, remote, publication

    async def coding(*args):
        _, product, _, _ = await verified(*args)
        return None, product

    async def verify(_task, given):
        assert given is product
        return accepted

    monkeypatch.setattr(module, "load_verified_github_result", verified)
    monkeypatch.setattr(module, "load_verified_coding_result", coding)
    monkeypatch.setattr(application, "verify", verify)
    return server, application, registry, provider, module, result, accepted, reads


@pytest.mark.parametrize("origin", ["https://github.example", "https://github.example:8443"])
def test_final_projection_links_resolve_and_are_read_only(final_context, monkeypatch, origin):
    server, application, registry, provider, _module, result, _accepted, _reads = final_context
    monkeypatch.setenv("CAYU_MAINTENANCE_GITHUB_WEB_ORIGIN", origin)
    result.pull_request.url = origin + "/fixture/maintenance/pull/7"

    async def scenario():
        async with client(server) as http:
            created = await http.post("/runs", headers=_A, json=_BODY)
            public_id = created.json()["id"]
            before = await application.app.task_store.list_tasks(TaskQuery())
            response = await http.get(
                f"/operator/runs/{public_id}/result?tenant=tenant-a", headers=_OP
            )
            assert response.status_code == 200, response.text
            data = response.json()
            assert data["outcome"] == "recorded_verified_delivery"
            assert data["observation"] == "durable_history"
            assert data["commit"] == result.head_commit
            assert data["pull_request"] == {"url": result.pull_request.url, "number": 7}
            for name in ("acceptance", "cost"):
                linked = await http.get(data[name]["href"], headers=_OP)
                assert linked.status_code == 200
                assert linked.json()["id"] == public_id
            assert data["cost"]["estimated_total"] is None
            assert data["cost"]["billing_completeness"] == "not_established"
            assert not provider.requests
            assert await application.app.task_store.list_tasks(TaskQuery()) == before
            identity = await registry.load_owned(tenant="tenant-a", public_id=public_id)
            assert identity.intent.subject not in response.text
            assert str(application.project_root) not in response.text

    asyncio.run(scenario())


@pytest.mark.parametrize("suffix", ["result", "acceptance"])
def test_auth_and_tenant_ownership_precede_readback(final_context, suffix):
    server, _app, _registry, _provider, _module, _result, _accepted, reads = final_context

    async def scenario():
        async with client(server) as http:
            created = await http.post("/runs", headers=_A, json=_BODY)
            path = f"/operator/runs/{created.json()['id']}/{suffix}"
            for headers in ({}, _A, _B):
                assert (
                    await http.get(path + "?tenant=tenant-a", headers=headers)
                ).status_code == 401
            assert (await http.get(path + "?tenant=tenant-b", headers=_OP)).status_code == 404
            assert (await http.get(path + "?tenant=a&tenant=b", headers=_OP)).status_code == 422
            assert reads == []

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "fault",
    [
        "http",
        "credentials",
        "query",
        "fragment",
        "path",
        "uppercase",
        "bad-port",
        "empty-label",
        "missing-origin",
        "wrong-url",
        "wrong-number",
        "wrong-repository",
        "acceptance",
        "store",
        "pending",
        "cost",
        "hostile-acceptance",
        "acceptance-tuple",
        "other-host",
    ],
)
def test_unsafe_or_unavailable_final_evidence_never_publishes_success(
    final_context, monkeypatch, caplog, capsys, fault
):
    server, application, _registry, _provider, module, result, accepted, _reads = final_context
    canary = "private-final-canary"
    origins = {
        "http": "http://github.example",
        "credentials": f"https://user:{canary}@github.example",
        "query": f"https://github.example?{canary}",
        "fragment": f"https://github.example#{canary}",
        "path": f"https://github.example/{canary}",
        "uppercase": "https://GitHub.example",
        "bad-port": "https://github.example:000443",
        "empty-label": "https://github..example",
    }
    if fault in origins:
        monkeypatch.setenv("CAYU_MAINTENANCE_GITHUB_WEB_ORIGIN", origins[fault])
    elif fault == "missing-origin":
        monkeypatch.delenv("CAYU_MAINTENANCE_GITHUB_WEB_ORIGIN")
    elif fault == "wrong-url":
        result.pull_request.url = "javascript:" + canary
    elif fault == "other-host":
        result.pull_request.url = "https://other.example/fixture/maintenance/pull/7"
    elif fault == "wrong-number":
        result.pull_request.number = True
    elif fault == "wrong-repository":
        result.repository_id = canary
    elif fault in {"acceptance", "hostile-acceptance", "acceptance-tuple"}:

        class Hostile:
            def __repr__(self):
                return canary

        async def invalid(*args):
            if fault == "hostile-acceptance":
                return replace(accepted, corpus_fingerprint=Hostile())
            if fault == "acceptance-tuple":
                return replace(accepted, result_digest="f" * 64)
            return replace(accepted, corpus_fingerprint=canary)

        monkeypatch.setattr(application, "verify", invalid)
    else:

        async def unavailable(*args):
            if fault == "pending":
                raise module.MaintenanceTaskConflict()
            raise OSError(canary)

        monkeypatch.setattr(
            module,
            "inspect_cost_evidence" if fault == "cost" else "load_verified_github_result",
            unavailable,
        )

    async def scenario():
        async with client(server) as http:
            created = await http.post("/runs", headers=_A, json=_BODY)
            response = await http.get(
                f"/operator/runs/{created.json()['id']}/result?tenant=tenant-a", headers=_OP
            )
            expected = (
                409
                if fault
                in {
                    "wrong-url",
                    "wrong-number",
                    "wrong-repository",
                    "acceptance",
                    "pending",
                    "hostile-acceptance",
                    "acceptance-tuple",
                    "other-host",
                }
                else 503
            )
            assert response.status_code == expected
            assert response.json() == {
                "detail": "Maintenance result is not verified."
                if expected == 409
                else "Maintenance result unavailable."
            }

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        asyncio.run(scenario())
    assert all(canary not in str(item.message) for item in captured)
    output = capsys.readouterr()
    assert canary not in caplog.text + output.out + output.err


@pytest.mark.parametrize("boundary", ["readback", "verify", "cost"])
def test_final_request_cancellation_preserves_signal(final_context, monkeypatch, boundary):
    server, application, _registry, provider, module, _result, _accepted, _reads = final_context

    async def scenario():
        entered = asyncio.Event()

        async def blocked(*args):
            entered.set()
            await asyncio.Event().wait()

        target, name = {
            "readback": (module, "load_verified_github_result"),
            "verify": (application, "verify"),
            "cost": (module, "inspect_cost_evidence"),
        }[boundary]
        monkeypatch.setattr(target, name, blocked)
        async with client(server) as http:
            created = await http.post("/runs", headers=_A, json=_BODY)
            before = await application.app.task_store.list_tasks(TaskQuery())
            owner = asyncio.create_task(
                http.get(
                    f"/operator/runs/{created.json()['id']}/result?tenant=tenant-a", headers=_OP
                )
            )
            try:
                await asyncio.wait_for(entered.wait(), 5)
                owner.cancel("final-disconnected")
                with pytest.raises(asyncio.CancelledError, match="final-disconnected"):
                    await owner
                assert owner.cancelled() and owner.cancelling() == 1
            finally:
                if not owner.done():
                    owner.cancel()
                await asyncio.gather(owner, return_exceptions=True)
            assert await application.app.task_store.list_tasks(TaskQuery()) == before
            assert not provider.requests

    asyncio.run(scenario())
