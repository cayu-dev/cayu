"""HTTP supervision-event contracts."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")
pytest.importorskip("httpx")

import asyncio
import base64
import json

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from tests.runtime.runtime_session_support import collect_tool_approval_events
from tests.runtime.runtime_supervision_support import (
    ARGUMENT_CANARY,
    WINDOW_END,
    WINDOW_START,
    approval_app,
    pause_for_approval,
    resolve_approval,
)

from cayu._event_schema import private_event_linkage_value
from cayu.approvals.tools import ToolApprovalDecision
from cayu.events import EventType
from cayu.providers.base import ModelStreamEvent
from cayu.resource_access import ResourceAccessPolicy
from cayu.runtime._event_projection import public_event_id, public_event_linkage_id
from cayu.server import ServerConfig, create_server
from cayu.server.auth import AuthContext
from cayu.server.resource_access import create_resource_router
from cayu.sessions import _pending_approval_reader as pending_approval_reader
from cayu.sessions._access_rules import SessionAccessRule, SessionAccessScope, SessionAccessSelector
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.memory import InMemorySessionStore

_WINDOW = {"start_at": WINDOW_START.isoformat(), "end_at": WINDOW_END.isoformat()}


def _authenticate(request: Request) -> AuthContext:
    return AuthContext(subject="auditor", tenant=request.headers.get("x-tenant") or None)


def _tenant_server(store):
    app = approval_app(store, sessions=3)
    app.get_provider().event_batches[0][0] = ModelStreamEvent.tool_call(
        id="call_1", name="side_effect", arguments={"value": ARGUMENT_CANARY}
    )
    client = TestClient(create_server(app, config=ServerConfig.protected(_authenticate)))
    return app, client


def _run_over_http(client, session_id: str, tenant: str) -> None:
    with client.stream(
        "POST",
        "/api/run",
        headers={"x-tenant": tenant},
        json={"prompt": "use the tool", "session_id": session_id},
    ) as response:
        assert response.status_code == 200
        list(response.iter_lines())


async def _approve(app, store, session_id: str) -> str:
    pending = pending_approval_reader.pending_approval_from_checkpoint(
        await store.load_checkpoint(session_id)
    )
    assert pending is not None
    await collect_tool_approval_events(
        app,
        resolve_approval(
            session_id, pending, ToolApprovalDecision.APPROVE, reason="checked by auditor"
        ),
    )
    return pending.approval_id


def test_tenant_identity_sees_only_its_sessions_and_operator_sees_all():
    store = InMemorySessionStore()
    app, client = _tenant_server(store)
    approval_ids = {}
    for session_id, tenant in (("sess_http_acme", "acme"), ("sess_http_globex", "globex")):
        _run_over_http(client, session_id, tenant)
        approval_ids[session_id] = asyncio.run(_approve(app, store, session_id))

    async def unattributed():
        await pause_for_approval(app, store, "sess_http_unattributed")

    asyncio.run(unattributed())
    approval_ids["sess_http_unattributed"] = asyncio.run(
        _approve(app, store, "sess_http_unattributed")
    )

    def query(tenant: str | None):
        headers = {} if tenant is None else {"x-tenant": tenant}
        response = client.post("/api/supervision-events/query", headers=headers, json=_WINDOW)
        assert response.status_code == 200, response.text
        assert response.headers["cache-control"] == "private, no-store"
        return response.json()

    acme, globex, operator = query("acme"), query("globex"), query(None)
    expose = app.project_session_id_for_exposure

    assert [event["session_id"] for event in acme["events"]] == [expose("sess_http_acme")]
    assert [event["session_id"] for event in globex["events"]] == [expose("sess_http_globex")]
    assert {event["session_id"] for event in operator["events"]} == {
        expose("sess_http_acme"),
        expose("sess_http_globex"),
        expose("sess_http_unattributed"),
    }
    body = str(operator)
    for approval_id in approval_ids.values():
        assert approval_id not in body
    assert ARGUMENT_CANARY not in body
    assert "claims" not in body
    record = acme["events"][0]
    assert record["kind"] == "approve"
    assert record["reason"] == "checked by auditor"
    assert record["tool_call_count"] == 1
    assert record["approval_id"].endswith(":approval_id")
    assert record["approval_id"].startswith(record["event_id"])


def test_linkage_aliases_name_the_anchor_event():
    store = InMemorySessionStore()
    app, client = _tenant_server(store)
    _run_over_http(client, "sess_http_alias", "acme")
    approval_id = asyncio.run(_approve(app, store, "sess_http_alias"))

    response = client.post(
        "/api/supervision-events/query", headers={"x-tenant": "acme"}, json=_WINDOW
    )
    record = response.json()["events"][0]
    approved = asyncio.run(
        store.query_events(
            EventQuery(session_id="sess_http_alias", event_type=EventType.TOOL_CALL_APPROVED)
        )
    )

    assert approved[0].event.payload["approval_id"] == approval_id
    assert record["event_id"] == public_event_id(approved[0].sequence)
    assert record["approval_id"] == public_event_linkage_id(approved[0].sequence, "approval_id")
    assert record["tool_round_id"] == public_event_linkage_id(approved[0].sequence, "tool_round_id")


def test_store_without_supervision_events_returns_501():
    class NoSupervisionStore(InMemorySessionStore):
        supports_supervision_events = False

    store = NoSupervisionStore()
    app = approval_app(store, sessions=1)
    client = TestClient(create_server(app, config=ServerConfig.protected(_authenticate)))

    response = client.post("/api/supervision-events/query", json=_WINDOW)

    assert response.status_code == 501


def test_cursor_from_another_query_is_rejected_without_echo():
    store = InMemorySessionStore()
    _app, client = _tenant_server(store)
    first = client.post(
        "/api/supervision-events/query", json={**_WINDOW, "kinds": ["approve"]}
    ).json()

    response = client.post(
        "/api/supervision-events/query",
        json={**_WINDOW, "kinds": ["deny"], "cursor": first["next_cursor"]},
    )

    assert response.status_code == 422
    assert first["next_cursor"] not in response.text


def test_content_cannot_be_requested_over_http():
    store = InMemorySessionStore()
    _app, client = _tenant_server(store)

    response = client.post(
        "/api/supervision-events/query", json={**_WINDOW, "include_content": True}
    )

    assert response.status_code == 422


class _AcmePolicy(ResourceAccessPolicy):
    authority = "tests:supervision-http-acme:v1"

    async def resolve(self, subject):
        rule = SessionAccessRule(
            selectors=(SessionAccessSelector(key="organization", values=("acme",)),)
        )
        return SessionAccessScope(read=(rule,), execute=(rule,), inspect_state=(rule,))


def test_resource_router_returns_only_admitted_sessions():
    store = InMemorySessionStore()
    app = approval_app(store, sessions=2, resource_access_policy=_AcmePolicy())

    async def run():
        for session_id, organization in (("sess_rr_acme", "acme"), ("sess_rr_globex", "globex")):
            pending = await pause_for_approval(
                app, store, session_id, labels={"organization": organization}
            )
            await collect_tool_approval_events(
                app, resolve_approval(session_id, pending, ToolApprovalDecision.APPROVE)
            )

    asyncio.run(run())
    api = FastAPI()

    async def authenticate(_request: Request) -> str:
        return "auditor"

    api.include_router(create_resource_router(app, authenticate=authenticate))
    client = TestClient(api)

    response = client.post("/resources/supervision-events/query", json=_WINDOW)
    bad_cursor = client.post(
        "/resources/supervision-events/query", json={**_WINDOW, "cursor": "not*a*cursor"}
    )

    assert response.status_code == 200, response.text
    assert [event["session_id"] for event in response.json()["events"]] == ["sess_rr_acme"]
    assert bad_cursor.status_code == 422


def test_page_over_its_byte_limit_returns_413():
    store = InMemorySessionStore()
    app, client = _tenant_server(store)
    _run_over_http(client, "sess_http_bytes", "acme")
    asyncio.run(_approve(app, store, "sess_http_bytes"))

    response = client.post("/api/supervision-events/query", json={**_WINDOW, "max_bytes": 64})

    assert response.status_code == 413
    assert response.headers["cache-control"] == "private, no-store"


def test_every_alias_resolves_and_cursors_stay_with_their_tenant():
    store = InMemorySessionStore()
    app, client = _tenant_server(store)
    _run_over_http(client, "sess_http_resolve_a", "acme")
    asyncio.run(_approve(app, store, "sess_http_resolve_a"))
    _run_over_http(client, "sess_http_resolve_b", "acme")
    client.post(
        "/api/sessions/sess_http_resolve_b/interrupt",
        headers={"x-tenant": "acme"},
        json={"reason": "stop"},
    )

    page = client.post(
        "/api/supervision-events/query", headers={"x-tenant": "acme"}, json=_WINDOW
    ).json()
    records = asyncio.run(store.query_events(EventQuery(limit=1000)))
    by_sequence = {record.sequence: record for record in records}
    aliases = 0
    for event in page["events"]:
        anchor = by_sequence[int(event["event_id"].rsplit("_", 1)[-1])]
        for field in (
            "model_step_id",
            "model_attempt_id",
            "tool_round_id",
            "approval_id",
            "input_id",
            "queue_id",
            "interruption_request_id",
        ):
            if event[field] is None:
                continue
            aliases += 1
            assert event[field] == public_event_linkage_id(anchor.sequence, field)
            assert private_event_linkage_value(anchor.event, field_name=field) is not None
    foreign = client.post(
        "/api/supervision-events/query",
        headers={"x-tenant": "globex"},
        json={**_WINDOW, "cursor": page["next_cursor"]},
    )
    operator = client.post(
        "/api/supervision-events/query",
        json={**_WINDOW, "cursor": page["next_cursor"]},
    )

    assert {event["kind"] for event in page["events"]} >= {"approve"}
    assert aliases > 0
    assert foreign.status_code == 422
    assert operator.status_code == 422


def test_cursor_with_a_non_ascii_digest_returns_422_without_echo():
    store = InMemorySessionStore()
    _app, client = _tenant_server(store)
    cursor = base64.urlsafe_b64encode(
        json.dumps(
            {"after_sequence": 0, "query_sha256": "é", "version": 1},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).decode()

    response = client.post("/api/supervision-events/query", json={**_WINDOW, "cursor": cursor})

    assert response.status_code == 422
    assert cursor not in response.text
