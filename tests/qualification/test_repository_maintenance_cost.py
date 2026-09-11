"""Generated HTTP cost evidence uses native causal accounting, not fixture totals."""

import asyncio
import warnings
from decimal import Decimal

import pytest

from cayu import (
    Event,
    EventType,
    Message,
    ModelPrice,
    PriceBook,
    RunRequest,
    SessionIdentity,
    TaskQuery,
)
from tests.qualification.test_repository_maintenance_http import _A, _B, _BODY, _OP, client
from tests.qualification.test_repository_maintenance_http import host as host
from tests.qualification.test_repository_maintenance_request import consumer as consumer
from tests.qualification.test_repository_maintenance_request import project as project


async def _session(app, session_id, causal_id):
    price = app.budget_policy.limits[0].pricing.prices[0]
    await app.session_store.create(
        RunRequest(
            agent_name="coding",
            session_id=session_id,
            causal_budget_id=causal_id,
            messages=[Message.text("user", "private-cost-instruction")],
        ),
        identity=SessionIdentity(provider_name=price.provider_name, model=price.model),
    )


async def _completion(app, session_id, tokens):
    price = app.budget_policy.limits[0].pricing.prices[0]
    await app.session_store.append_event(
        session_id,
        Event(
            type=EventType.MODEL_COMPLETED,
            session_id=session_id,
            payload={
                "usage_metrics": {
                    "provider_name": price.provider_name,
                    "model": price.model,
                    "input_tokens": tokens,
                    "output_tokens": 0,
                    "total_tokens": tokens,
                }
            },
        ),
    )


@pytest.mark.parametrize("consumer", ["memory", "sqlite"], indirect=True)
@pytest.mark.parametrize("mode", ["absent", "empty", "priced", "unpriced-hosted"])
def test_cost_view_reads_actual_causal_evidence(host, mode):
    server, application, registry, provider, _auth = host
    app = application.app

    async def scenario():
        async with client(server) as http:
            created = await http.post("/runs", headers=_A, json=_BODY)
            identity = await registry.load_owned(tenant="tenant-a", public_id=created.json()["id"])
            root = identity.workflow_session_id
            if mode != "absent":
                await _session(app, root, root)
            if mode == "priced":
                await _session(app, identity.session_id, root)
                await _completion(app, root, 2)
                await _completion(app, identity.session_id, 3)
                await _session(app, "unrelated-session", "unrelated-cohort")
                await _completion(app, "unrelated-session", 100)
            if mode == "unpriced-hosted":
                await app.session_store.append_event(
                    root,
                    Event(
                        type=EventType.MODEL_HOSTED_TOOL_CALL,
                        session_id=root,
                        payload={
                            "tool_type": "web_search",
                            "call_id": "private-hosted-call",
                            "status": "completed",
                            "provider_name": "private-unpriced-provider",
                            "model": "private-unpriced-model",
                            "model_attempt_id": "private-attempt",
                        },
                    ),
                )
            before = await app.task_store.list_tasks(TaskQuery())
            url = f"/operator/runs/{identity.public_id}/cost?tenant=tenant-a"
            response = await http.get(url, headers=_OP)
            assert response.status_code == 200, response.text
            data = response.json()
            assert data["id"] == identity.public_id
            assert data["basis"] == "recorded_runtime_events"
            assert data["pricing_basis"] == "current_configured_policy"
            assert data["billing_completeness"] == "not_established"
            assert len(data["pricing_fingerprint"]) == 71
            assert data["currency"] == "USD"
            if mode == "absent":
                assert data["evidence"] == "not_recorded"
                assert data["estimated_total"] is None
                assert data["pricing_coverage"] == "not_available"
                assert data["session_count"] is None
            else:
                assert data["evidence"] == "recorded"
                assert Decimal(data["estimated_total"]) == (5 if mode == "priced" else 0)
                assert data["session_count"] == (2 if mode == "priced" else 1)
                assert data["model_steps"] == (2 if mode == "priced" else 0)
                assert (
                    data["observed_line_items"]
                    == {"empty": 0, "priced": 2, "unpriced-hosted": 1}[mode]
                )
                assert data["unpriced_line_items"] == (1 if mode == "unpriced-hosted" else 0)
                assert (
                    data["pricing_coverage"]
                    == {
                        "empty": "no_cost_observations",
                        "priced": "all_observed_items_priced",
                        "unpriced-hosted": "unpriced_observations",
                    }[mode]
                )
            assert (await http.get(url, headers=_OP)).json() == data
            assert await app.task_store.list_tasks(TaskQuery()) == before
            assert not provider.requests
            for secret in (
                root,
                identity.intent.subject,
                "private-",
                str(application.project_root),
            ):
                assert secret not in response.text

    asyncio.run(scenario())


def test_cost_authentication_precedes_accounting(host, monkeypatch):
    server, application, _registry, _provider, _auth = host

    async def forbidden(*args, **kwargs):
        pytest.fail("Unauthorized request reached accounting")

    async def scenario():
        async with client(server) as http:
            created = await http.post("/runs", headers=_A, json=_BODY)
            url = f"/operator/runs/{created.json()['id']}/cost"
            monkeypatch.setattr(application.app, "get_causal_budget_cost", forbidden)
            for headers in ({}, _A, _B):
                assert (
                    await http.get(url + "?tenant=tenant-a", headers=headers)
                ).status_code == 401
            for query in ("", "?tenant=a&tenant=b", "?tenant=%20a"):
                assert (await http.get(url + query, headers=_OP)).status_code == 422
            assert (await http.get(url + "?tenant=tenant-b", headers=_OP)).status_code == 404

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "failure",
    ["store", "wrong-type", "wrong-causal", "currency", "mutated", "boolean", "nonfinite"],
)
def test_invalid_cost_is_sanitized_without_serializing_it(
    host, monkeypatch, caplog, capsys, failure
):
    server, application, registry, _provider, _auth = host
    app = application.app
    canary = "private-cost-canary"

    class Hostile:
        def __repr__(self):
            return canary

    async def scenario():
        async with client(server) as http:
            created = await http.post("/runs", headers=_A, json=_BODY)
            identity = await registry.load_owned(tenant="tenant-a", public_id=created.json()["id"])
            await _session(app, identity.workflow_session_id, identity.workflow_session_id)
            original = app.get_causal_budget_cost

            async def broken(*args, **kwargs):
                if failure == "store":
                    raise OSError(canary)
                if failure == "wrong-type":
                    return Hostile()
                summary = await original(*args, **kwargs)
                field, value = {
                    "wrong-causal": ("causal_budget_id", canary),
                    "currency": ("currency", "EUR"),
                    "mutated": ("model_steps", Hostile()),
                    "boolean": ("model_steps", True),
                    "nonfinite": ("total_cost", Decimal("Infinity")),
                }[failure]
                return summary.model_copy(update={field: value})

            monkeypatch.setattr(app, "get_causal_budget_cost", broken)
            response = await http.get(
                f"/operator/runs/{identity.public_id}/cost?tenant=tenant-a", headers=_OP
            )
            assert response.status_code == 503
            assert response.json() == {"detail": "Maintenance cost unavailable."}

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        asyncio.run(scenario())
    assert all(canary not in str(item.message) for item in captured)
    output = capsys.readouterr()
    assert canary not in caplog.text + output.out + output.err


def test_cost_request_cancellation_keeps_original_signal(host, monkeypatch):
    server, application, _registry, provider, _auth = host

    async def scenario():
        async with client(server) as http:
            created = await http.post("/runs", headers=_A, json=_BODY)
            before = await application.app.task_store.list_tasks(TaskQuery())
            entered = asyncio.Event()

            async def blocked(*args, **kwargs):
                entered.set()
                await asyncio.Event().wait()

            monkeypatch.setattr(application.app, "get_causal_budget_cost", blocked)
            request = asyncio.create_task(
                http.get(f"/operator/runs/{created.json()['id']}/cost?tenant=tenant-a", headers=_OP)
            )
            try:
                await asyncio.wait_for(entered.wait(), 5)
                request.cancel("cost-disconnected")
                with pytest.raises(asyncio.CancelledError, match="cost-disconnected"):
                    await request
                assert request.cancelled() and request.cancelling() == 1
            finally:
                if not request.done():
                    request.cancel()
                await asyncio.gather(request, return_exceptions=True)
            assert await application.app.task_store.list_tasks(TaskQuery()) == before
            assert not provider.requests

    asyncio.run(scenario())


def test_cost_captures_pricing_before_accounting_and_identifies_repricing(host, monkeypatch):
    server, application, registry, _provider, _auth = host
    app = application.app

    async def scenario():
        async with client(server) as http:
            created = await http.post("/runs", headers=_A, json=_BODY)
            identity = await registry.load_owned(tenant="tenant-a", public_id=created.json()["id"])
            await _session(app, identity.workflow_session_id, identity.workflow_session_id)
            await _completion(app, identity.workflow_session_id, 2)
            original = app.get_causal_budget_cost
            price = app.budget_policy.limits[0].pricing.prices[0]
            changed = PriceBook(
                prices=(
                    ModelPrice.fixed(
                        provider_name=price.provider_name,
                        model=price.model,
                        input_per_million=Decimal("2000000"),
                        output_per_million=Decimal("1000000"),
                    ),
                )
            )

            async def replace_policy(causal_id, captured, **kwargs):
                app.budget_policy.limits[0].pricing = changed
                assert captured is not changed
                return await original(causal_id, captured, **kwargs)

            monkeypatch.setattr(app, "get_causal_budget_cost", replace_policy)
            url = f"/operator/runs/{identity.public_id}/cost?tenant=tenant-a"
            first = await http.get(url, headers=_OP)
            assert first.status_code == 200
            assert Decimal(first.json()["estimated_total"]) == 2
            monkeypatch.setattr(app, "get_causal_budget_cost", original)
            second = await http.get(url, headers=_OP)
            assert second.status_code == 200
            assert Decimal(second.json()["estimated_total"]) == 4
            assert first.json()["pricing_fingerprint"] != second.json()["pricing_fingerprint"]
            assert first.json()["pricing_basis"] == second.json()["pricing_basis"]
            assert second.json()["billing_completeness"] == "not_established"

    asyncio.run(scenario())
