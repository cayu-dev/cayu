"""Authenticated HTTP lifecycle for durable tool authorization grants."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi import HTTPException, Request
from fastapi.testclient import TestClient
from tests.runtime.authorization_grant_support import Harness, _command_turn
from tests.server.server_test_support import _sse_frames

from cayu.approvals import ProcessCommandGrantMatcher
from cayu.server import AuthContext, ServerConfig, create_server
from cayu.server.authorization_grants import ToolAuthorizationGrantServer


def _auth(request: Request) -> AuthContext:
    subject = request.headers.get("x-test-subject")
    if not subject:
        raise HTTPException(status_code=401, detail="unauthenticated")
    return AuthContext(subject=subject, tenant="acme")


async def _owner_reads(principal: AuthContext, grant) -> bool:
    return principal.subject in {"owner", "intruder"}


def _server(harness: Harness):
    return create_server(
        harness.app,
        config=ServerConfig.protected(_auth),
        tool_authorization_grants=ToolAuthorizationGrantServer(
            harness.service, authorize_read=_owner_reads
        ),
    )


def _client(harness: Harness) -> TestClient:
    return TestClient(_server(harness))


def _approve_body(pending, **grant) -> dict:
    return {
        "session_id": "s1",
        "approval_id": pending.approval_id,
        "tool_round_id": pending.tool_round_id,
        "tool_call_id": pending.tool_call_id,
        "reason": "routine",
        "grant": {"scope": "session", **grant},
    }


def test_grant_routes_require_authentication_and_trusted_actor(tmp_path) -> None:
    harness = Harness(tmp_path, turns=_command_turn("call_1", ["ls"]))
    asyncio.run(harness.run("s1"))
    pending = asyncio.run(harness.pending("s1"))
    client = _client(harness)
    path = "/api/tool-authorization-grants/approve"

    assert client.get("/api/tool-authorization-grants").status_code == 401
    assert client.post(path, json=_approve_body(pending)).status_code == 401
    spoofed = {**_approve_body(pending), "resolved_by": {"subject": "owner"}}
    assert (
        client.post(path, json=spoofed, headers={"x-test-subject": "intruder"}).status_code == 422
    )
    smuggled = _approve_body(pending)
    smuggled["grant"]["tool_name"] = "other"
    assert client.post(path, json=smuggled, headers={"x-test-subject": "owner"}).status_code == 422

    denied = client.post(path, json=_approve_body(pending), headers={"x-test-subject": "intruder"})
    assert denied.status_code == 403
    assert asyncio.run(harness.service.list_grants()) == ()
    assert asyncio.run(harness.pending("s1")).approval_id == pending.approval_id
    assert harness.tool.calls == []


def test_grant_lifecycle_over_http(tmp_path) -> None:
    turns = _command_turn("call_1", ["ls"]) + _command_turn("call_2", ["ls"])
    harness = Harness(tmp_path, turns=turns)
    asyncio.run(harness.run("s1"))
    pending = asyncio.run(harness.pending("s1"))
    client = _client(harness)
    owner = {"x-test-subject": "owner"}

    with client.stream(
        "POST",
        "/api/tool-authorization-grants/approve",
        json=_approve_body(pending, sampled_review_rate_bps=0, justification="unchanged"),
        headers=owner,
    ) as response:
        assert response.status_code == 200
        types = [frame["data"]["type"] for frame in _sse_frames(response) if "data" in frame]
    assert "tool.call.approved" in types
    assert harness.tool.calls == [{"argv": ["ls"]}]

    listed = client.get(
        "/api/tool-authorization-grants", params={"status": "active"}, headers=owner
    )
    assert listed.status_code == 200
    (view,) = listed.json()["grants"]
    grant_id = view["grant"]["grant_id"]
    assert view["effective_status"] == "active"
    assert view["grant"]["created_by"] == {
        "subject": "owner",
        "tenant": "acme",
        "source": "http_auth",
    }
    assert view["grant"]["origin"]["approval_id"] == pending.approval_id
    assert view["grant"]["matcher"] == {
        "kind": "process_command",
        "version": 1,
        "argv": ["ls"],
        "cwd": None,
        "private_inputs_digest": ProcessCommandGrantMatcher.from_arguments(
            {"argv": ["ls"]}
        ).private_inputs_digest,
    }

    later = asyncio.run(harness.run("s1"))
    assert "tool.call.approval_requested" not in [str(event.type) for event in later]
    uses = client.get(f"/api/tool-authorization-grants/{grant_id}/uses", headers=owner).json()
    assert [use["decision"] for use in uses["uses"]] == ["allowed"]

    forbidden = client.post(
        f"/api/tool-authorization-grants/{grant_id}/revoke",
        json={"reason": "not mine"},
        headers={"x-test-subject": "intruder"},
    )
    assert forbidden.status_code == 403
    revoked = client.post(
        f"/api/tool-authorization-grants/{grant_id}/revoke",
        json={"reason": "sampled errors"},
        headers=owner,
    )
    assert revoked.status_code == 200
    assert revoked.json()["effective_status"] == "revoked"
    assert client.get("/api/tool-authorization-grants/missing", headers=owner).status_code == 404


def test_grant_routes_are_declared_in_openapi_only_when_configured(tmp_path) -> None:
    harness = Harness(tmp_path, turns=_command_turn("call_1", ["ls"]))
    paths = _server(harness).openapi()["paths"]
    assert {
        "/api/tool-authorization-grants",
        "/api/tool-authorization-grants/approve",
        "/api/tool-authorization-grants/{grant_id}",
        "/api/tool-authorization-grants/{grant_id}/uses",
        "/api/tool-authorization-grants/{grant_id}/revoke",
        "/api/tool-authorization-grants/{grant_id}/reconcile",
    } <= set(paths)
    plain = create_server(harness.app, config=ServerConfig.protected(_auth))
    assert not any("tool-authorization-grants" in path for path in plain.openapi()["paths"])
    with pytest.raises(ValueError, match="authenticated access"):
        create_server(
            harness.app,
            config=ServerConfig.local_development(),
            tool_authorization_grants=ToolAuthorizationGrantServer(
                harness.service, authorize_read=_owner_reads
            ),
        )


_GRANT_SNAPSHOT = Path(__file__).parent / "snapshots" / "tool-authorization-grants-openapi.json"


def _grant_contract(schema: dict) -> dict:
    """The grant routes and only the component schemas they reference."""

    from tests.server.test_server_contract import _openapi_contract_summary

    summary = _openapi_contract_summary(schema)
    paths = {
        path: item
        for path, item in summary["paths"].items()
        if path.startswith("/api/tool-authorization-grants")
    }
    components = summary["components"]
    selected: dict = {}
    pending = [paths]
    while pending:
        node = pending.pop()
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/components/schemas/"):
                name = ref.rsplit("/", 1)[1]
                if name not in selected:
                    selected[name] = components[name]
                    pending.append(components[name])
            pending.extend(node.values())
        elif isinstance(node, list):
            pending.extend(node)
    return {"paths": paths, "components": dict(sorted(selected.items()))}


def test_grant_routes_openapi_contract_matches_snapshot(tmp_path) -> None:
    harness = Harness(tmp_path, turns=_command_turn("call_1", ["ls"]))
    schema = _server(harness).openapi()

    assert _grant_contract(schema) == json.loads(_GRANT_SNAPSHOT.read_text())


def test_grant_listing_pages_over_http_and_refuses_task_fields(tmp_path) -> None:
    from tests.runtime.test_tool_authorization_grant_hardening import _seed_expired_grants

    harness = Harness(tmp_path, turns=_command_turn("call_1", ["ls"]))
    asyncio.run(harness.run("s1"))
    pending = asyncio.run(harness.pending("s1"))
    client = _client(harness)
    owner = {"x-test-subject": "owner"}
    with_task = {**_approve_body(pending), "task_worker_id": "worker-1"}
    assert (
        client.post(
            "/api/tool-authorization-grants/approve", json=with_task, headers=owner
        ).status_code
        == 422
    )
    with client.stream(
        "POST",
        "/api/tool-authorization-grants/approve",
        json=_approve_body(pending),
        headers=owner,
    ) as response:
        assert response.status_code == 200
        list(response.iter_lines())
    asyncio.run(_seed_expired_grants(harness, count=4))

    seen: list[str] = []
    params: dict = {"limit": 2}
    while True:
        body = client.get("/api/tool-authorization-grants", params=params, headers=owner).json()
        seen.extend(view["grant"]["grant_id"] for view in body["grants"])
        if body["next"] is None:
            break
        params = {"limit": 2, **body["next"]}

    assert len(seen) == len(set(seen)) == 5


def test_grant_uses_page_over_http_by_cursor(tmp_path) -> None:
    from cayu.approvals import ToolAuthorizationGrantUse, ToolAuthorizationGrantUseDecision

    turns = _command_turn("call_1", ["ls"])
    harness = Harness(tmp_path, turns=turns)
    asyncio.run(harness.run("s1"))
    client = _client(harness)
    owner = {"x-test-subject": "owner"}
    pending = asyncio.run(harness.pending("s1"))
    with client.stream(
        "POST", "/api/tool-authorization-grants/approve", json=_approve_body(pending), headers=owner
    ) as response:
        list(response.iter_lines())
    (grant,) = asyncio.run(harness.service.list_grants())
    for index in range(5):
        asyncio.run(
            harness.grants.record_use(
                ToolAuthorizationGrantUse(
                    grant_id=grant.grant_id,
                    policy_id=grant.policy_id,
                    scope=grant.scope,
                    session_id="s1",
                    session_instance_id=grant.origin.session_instance_id,
                    tool_round_id=f"round-{index}",
                    tool_call_id="call",
                    tool_name="exec_command",
                    arguments_digest="sha256:args",
                    decision=ToolAuthorizationGrantUseDecision.ALLOWED,
                    sample_bucket=9_999,
                    sampled_review_rate_bps=0,
                    recorded_at=grant.created_at,
                )
            )
        )

    seen: list[str] = []
    params: dict = {"limit": 2}
    while True:
        body = client.get(
            f"/api/tool-authorization-grants/{grant.grant_id}/uses", params=params, headers=owner
        ).json()
        seen.extend(use["tool_round_id"] for use in body["uses"])
        if body["next_cursor"] is None:
            break
        params = {"limit": 2, "cursor": body["next_cursor"]}

    assert sorted(seen) == [f"round-{index}" for index in range(5)]
    assert len(seen) == len(set(seen))
