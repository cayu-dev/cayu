"""Actual authenticated HTTP-to-durable-request boundary, without native grant."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi import FastAPI
from tests.core.test_browser_control_authorization import Policy
from tests.core.test_browser_control_coordinator import coordinator, intent_for
from tests.core.test_browser_control_publisher import publication_fixture

from cayu.runtime._browser_control_authorization import BrowserControlPermissionDenied
from cayu.runtime._browser_control_input_tickets import BrowserInputTickets
from cayu.runtime._browser_control_publisher import BrowserControlPublisher
from cayu.runtime._browser_control_view_tickets import BrowserViewTickets
from cayu.runtime.browser_control import BrowserControlCheckpoint, BrowserControlPrincipal
from cayu.runtime.checkpoints import BROWSER_CONTROLS_CHECKPOINT_KEY
from cayu.server._browser_control_routes import (
    BrowserOperatorSessionTokens,
    create_browser_control_router,
)
from cayu.server.auth import BasicAuth


@pytest.mark.parametrize("allowed", [False, True])
def test_http_view_ticket_requires_policy_and_is_single_use(tmp_path, allowed):
    async def scenario():
        async with publication_fixture("sqlite", tmp_path) as (store, bootstrap):
            record = await BrowserControlPublisher(store).publish(bootstrap)
            owner = coordinator(store, Policy(allowed))
            now = [1.0]
            tickets = BrowserViewTickets(owner, clock=lambda: now[0])
            server = FastAPI()
            server.include_router(
                create_browser_control_router(
                    coordinator=owner,
                    auth=BasicAuth(username="operator", password="password", tenant="tenant"),
                    sessions=BrowserOperatorSessionTokens(b"k" * 32),
                    allowed_origin="https://operator.test",
                    view_tickets=tickets,
                )
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server),
                base_url="https://operator.test",
                auth=httpx.BasicAuth("operator", "password"),
            ) as client:
                response = await client.post("/browser-control/operator-session")
                token = response.json()["operator_session_token"]
                discovery = await client.get(
                    "/browser-control/sessions/session",
                    headers={"X-Cayu-Browser-Operator": token},
                )
                assert discovery.status_code == (200 if allowed else 403)
                if allowed:
                    assert discovery.headers["cache-control"] == "no-store"
                    descriptors = discovery.json()["browsers"]
                    assert len(descriptors) == 1
                    assert descriptors[0]["identity"] == record.identity.model_dump(mode="json")
                    assert set(descriptors[0]) == {
                        "identity",
                        "revision",
                        "control_epoch",
                        "state",
                        "sensitive_entry",
                        "sensitive_entry_pending",
                        "fresh_observation_required",
                        "owned_request",
                    }
                    assert descriptors[0]["owned_request"] is None
                else:
                    assert record.identity.browser_session_id not in discovery.text
                assert (await client.get("/browser-control/sessions/session")).status_code == 403
                payload = {
                    "identity": record.identity.model_dump(mode="json"),
                    "expected_record_revision": 1,
                    "page": {"page_id": "page", "revision": "revision", "control_epoch": 1},
                }
                before = await store.load_checkpoint("session")
                for expire in (False, True):
                    response = await client.post(
                        "/browser-control/view-ticket",
                        json=payload,
                        headers={"X-Cayu-Browser-Operator": token},
                    )
                    assert response.status_code == (200 if allowed else 403)
                    if allowed:
                        assert response.headers["cache-control"] == "no-store"
                        ticket = response.json()["ticket"]
                        assert ticket not in repr(tickets._pending)
                        if expire:
                            now[0] += 31
                        else:
                            viewer = tickets.consume(ticket)
                            assert viewer.principal == BrowserControlPrincipal(
                                subject="operator", tenant="tenant"
                            )
                            assert viewer.intent.identity == record.identity
                        with pytest.raises(BrowserControlPermissionDenied):
                            tickets.consume(ticket)
                assert await store.load_checkpoint("session") == before
            tickets.close()
            assert await owner.drain()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("allowed", [False, True])
@pytest.mark.parametrize("action", ["handback", "sensitive-entry"])
@pytest.mark.parametrize("consent", ["allow", "deny", "undecided"])
def test_authenticated_http_requires_policy_before_durable_takeover(
    backend, allowed, action, consent, tmp_path
):
    async def scenario():
        async with publication_fixture(backend, tmp_path) as (store, bootstrap):
            await BrowserControlPublisher(store).publish(bootstrap)
            server = FastAPI()
            policy = Policy(allowed)
            owner = coordinator(store, policy)
            input_tickets = BrowserInputTickets(owner)
            server.include_router(
                create_browser_control_router(
                    coordinator=owner,
                    auth=BasicAuth(username="operator", password="password", tenant="tenant"),
                    sessions=BrowserOperatorSessionTokens(b"k" * 32),
                    allowed_origin="https://operator.test",
                    input_tickets=input_tickets,
                )
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="https://operator.test"
            ) as client:
                response = await client.post("/browser-control/operator-session")
                assert response.status_code == 401
                client.auth = httpx.BasicAuth("operator", "password")
                response = await client.post("/browser-control/operator-session")
                assert response.status_code == 200
                assert response.headers["cache-control"] == "no-store"
                token = response.json()["operator_session_token"]
                assert not policy.requests  # Continuity is not browser permission.
                response = await client.post(
                    "/browser-control/takeover",
                    json=intent_for(bootstrap)
                    .model_copy(update={"checkpoint_consent": consent})
                    .model_dump(mode="json"),
                    headers={"X-Cayu-Browser-Operator": token},
                )
                assert response.status_code == (200 if allowed else 403)
                assert len(policy.requests) == 1
                assert policy.requests[0].principal.subject == "operator"
                assert policy.requests[0].principal.tenant == "tenant"
                checkpoint = await store.load_checkpoint("session")
                record = BrowserControlCheckpoint.model_validate(
                    checkpoint[BROWSER_CONTROLS_CHECKPOINT_KEY]
                ).records[0]
                assert record.state == ("takeover_requested" if allowed else "agent_controlled")
                assert record.lease_until_ms is None
                if allowed:
                    assert record.request is not None
                    assert record.checkpoint_consent == record.request.checkpoint_consent == consent
                    assert record.request.operator.subject == "operator"
                    assert record.request.operator.operator_session_id.startswith("bo_")
                    # Guest acquisition is covered separately by the ASGI/native
                    # protocol tests; seed its positive publication for this HTTP boundary.
                    acquired = await owner._publish_guest_acquisition(
                        expected=record, lease_until_ms=4000
                    )
                    handback = {
                        "identity": acquired.identity.model_dump(mode="json"),
                        "expected_record_revision": acquired.revision,
                        "expected_control_epoch": acquired.control_epoch,
                        "request_id": record.request.request_id,
                    }
                    other_token = (await client.post("/browser-control/operator-session")).json()[
                        "operator_session_token"
                    ]
                    for continuity, is_owner in ((token, True), (other_token, False)):
                        discovered = await client.get(
                            "/browser-control/sessions/session",
                            headers={"X-Cayu-Browser-Operator": continuity},
                        )
                        assert discovered.status_code == 200
                        assert discovered.headers["cache-control"] == "no-store"
                        own = discovered.json()["browsers"][0]["owned_request"]
                        if is_owner:
                            assert own == {
                                "request_id": record.request.request_id,
                                "expires_at_ms": record.request.expires_at_ms,
                                "maximum_until_ms": record.request.maximum_until_ms,
                                "lease_until_ms": 4000,
                                "pending_lease_until_ms": None,
                                "settled_input_sequence": 0,
                                "pending_input_sequence": None,
                                "checkpoint_consent": consent,
                            }
                        else:
                            assert own is None
                            assert record.request.request_id not in discovered.text
                    refused = await client.post(
                        f"/browser-control/{action}",
                        json=handback,
                        headers={"X-Cayu-Browser-Operator": other_token},
                    )
                    assert refused.status_code == 409
                    policy.allowed = False
                    before = await store.load_checkpoint("session")
                    denied = await client.post(
                        f"/browser-control/{action}",
                        json=handback,
                        headers={"X-Cayu-Browser-Operator": token},
                    )
                    assert denied.status_code == 403
                    assert await store.load_checkpoint("session") == before
                    policy.allowed = True
                    first = await client.post(
                        f"/browser-control/{action}",
                        json=handback,
                        headers={"X-Cayu-Browser-Operator": token},
                    )
                    expected_status = 202 if action == "sensitive-entry" else 200
                    assert first.status_code == expected_status
                    if action == "sensitive-entry":
                        assert first.json()["sensitive_entry_pending"] is True
                        assert first.json()["sensitive_entry"] is False
                        assert first.json()["state"] == "operator_controlled"
                    else:
                        assert first.json()["state"] == "handback_pending"
                    replay = await client.post(
                        f"/browser-control/{action}",
                        json=handback,
                        headers={"X-Cayu-Browser-Operator": token},
                    )
                    assert replay.status_code == expected_status and replay.json() == first.json()
                    _, current = await owner._load(acquired.identity)
                    assert current.checkpoint_consent == consent
                    if action == "sensitive-entry":
                        _, pending_sensitive = await owner._load(acquired.identity)
                        ready = await owner._publish_guest_sensitive_entry(
                            expected=pending_sensitive
                        )
                        input_intent = {
                            **handback,
                            "expected_record_revision": ready.revision,
                            "input_sequence": 1,
                            "page": ready.request.pages[0].model_dump(mode="json"),
                        }
                        before = await store.load_checkpoint("session")
                        refused = await client.post(
                            "/browser-control/input-ticket",
                            headers={"X-Cayu-Browser-Operator": token},
                            json={**input_intent, "text": "must-not-enter-http"},
                        )
                        assert refused.status_code == 422
                        assert "must-not-enter-http" not in refused.text
                        issued = await client.post(
                            "/browser-control/input-ticket",
                            headers={"X-Cayu-Browser-Operator": token},
                            json=input_intent,
                        )
                        assert issued.status_code == 200
                        assert issued.json()["expires_in_seconds"] == 10
                        assert issued.headers["cache-control"] == "no-store"
                        ticket = input_tickets.consume(issued.json()["ticket"])
                        assert ticket.intent.input_sequence == 1
                        assert ticket.principal.subject == "operator"
                        assert await store.load_checkpoint("session") == before

    asyncio.run(scenario())


def test_operator_continuity_is_bound_to_principal_expiry_and_key():
    now = [1000.0]
    codec = BrowserOperatorSessionTokens(b"k" * 32, clock=lambda: now[0])
    principal = BrowserControlPrincipal(subject="operator", tenant="tenant")
    token = codec.issue(principal)
    expected = codec.verify(token, principal)
    assert (
        BrowserOperatorSessionTokens(b"k" * 32, clock=lambda: now[0]).verify(token, principal)
        == expected
    )
    from cayu.runtime._browser_control_authorization import BrowserControlPermissionDenied

    for other in (
        BrowserControlPrincipal(subject="other", tenant="tenant"),
        BrowserControlPrincipal(subject="operator", tenant="other"),
    ):
        with pytest.raises(BrowserControlPermissionDenied):
            codec.verify(token, other)
    with pytest.raises(BrowserControlPermissionDenied):
        BrowserOperatorSessionTokens(b"x" * 32, clock=lambda: now[0]).verify(token, principal)
    now[0] = 4600
    with pytest.raises(BrowserControlPermissionDenied):
        codec.verify(token, principal)


def test_http_rejects_bad_origin_and_malformed_body_without_echo(tmp_path, capsys, caplog):
    async def scenario():
        async with publication_fixture("memory", tmp_path) as (store, bootstrap):
            await BrowserControlPublisher(store).publish(bootstrap)
            policy = Policy(True)
            server = FastAPI()
            server.include_router(
                create_browser_control_router(
                    coordinator=coordinator(store, policy),
                    auth=BasicAuth(username="operator", password="password"),
                    sessions=BrowserOperatorSessionTokens(b"k" * 32),
                    allowed_origin="https://operator.test",
                )
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server),
                base_url="https://operator.test",
                auth=httpx.BasicAuth("operator", "password"),
            ) as client:
                response = await client.post(
                    "/browser-control/operator-session", headers={"Origin": "https://other.test"}
                )
                assert response.status_code == 403
                token = (await client.post("/browser-control/operator-session")).json()[
                    "operator_session_token"
                ]
                response = await client.post(
                    "/browser-control/takeover",
                    json={"credential-canary": "private-canary"},
                    headers={"X-Cayu-Browser-Operator": token},
                )
                assert response.status_code == 422
                assert "canary" not in response.text
                assert not policy.requests
                output = capsys.readouterr()
                assert "canary" not in output.out + output.err + caplog.text

    asyncio.run(scenario())
