"""The shared-allocation routes list and release a crashed workflow run's sandbox."""

from __future__ import annotations

import asyncio
from hashlib import sha256
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi import HTTPException
from fastapi.testclient import TestClient
from tests.environments.test_durable_shared_allocations import _crash_after_plan, _World

from cayu.server import AuthenticatedAccess, ServerConfig, create_server
from cayu.server.auth import AuthContext
from cayu.server.proxy_identity import ControlPlaneRoles

_LIST = "/api/shared-allocations"
_RELEASE = "/api/shared-allocations/release"


def crashed_world(tmp_path: Path, run_id: str) -> _World:
    """A world whose first process ran the plan step of ``run_id`` and died."""

    world = _World(tmp_path)

    async def crash() -> None:
        try:
            await _crash_after_plan(world, run_id)
        finally:
            await world.close()

    asyncio.run(crash())
    world.stores.clear()
    return world


def test_an_operator_lists_and_discards_a_crashed_runs_allocation(tmp_path: Path) -> None:
    world = crashed_world(tmp_path, "run-http")
    server = create_server(world.app(), config=ServerConfig.local_development())
    with TestClient(server) as client:
        listed = client.get(_LIST)
        assert listed.status_code == 200, listed.text
        assert listed.headers["cache-control"] == "private, no-store"
        (allocation,) = listed.json()["allocations"]
        assert listed.json()["next_cursor"] is None
        assert allocation["workflow_run_id"] == "run-http"
        assert allocation["state"] == "live" and allocation["reconnectable"] is True
        assert allocation["run_state"] == "running"
        # No reconnect metadata or binding state in the response.
        assert "reconnect_metadata" not in listed.text and "binding_state" not in listed.text
        sandbox_name = world.sandboxes.requests[0].session_id
        assert sha256(sandbox_name.encode()).hexdigest()[:16] not in listed.text
        assert "sandbox" not in str(allocation["metadata"])
        only_run = client.get(_LIST, params={"workflow_run_id": "run-http"}).json()
        assert [row["allocation_id"] for row in only_run["allocations"]] == [
            allocation["allocation_id"]
        ]
        assert client.get(_LIST, params={"environment_name": "other"}).json()["allocations"] == []
        assert client.get(_LIST, params={"limit": 0}).status_code == 422

        body = {"allocation_id": allocation["allocation_id"], "mode": "discard"}
        refused = client.post(_RELEASE, json=body)
        assert refused.status_code == 409, refused.text
        assert "force" in refused.json()["detail"]

        released = client.post(_RELEASE, json={**body, "inactive_for_seconds": 0})
        assert released.status_code == 200, released.text
        answer = released.json()
        assert (answer["mode"], answer["forced"]) == ("discard", False)
        assert answer["allocation"]["allocation_id"] == allocation["allocation_id"]
        assert client.get(_LIST).json()["allocations"] == []
        again = client.post(_RELEASE, json={**body, "inactive_for_seconds": 0})
        assert again.status_code == 404
        assert client.post(_RELEASE, json={**body, "mode": "keep"}).status_code == 422
        assert client.post(_RELEASE, json={**body, "extra": True}).status_code == 422

    assert world.sandboxes.finalizations == []
    assert len(world.sandboxes.disposed) == 1


def test_a_copy_back_through_the_route_reaches_the_source(tmp_path: Path) -> None:
    world = crashed_world(tmp_path, "run-copy-http")
    server = create_server(world.app(), config=ServerConfig.local_development())
    with TestClient(server) as client:
        (allocation,) = client.get(_LIST).json()["allocations"]
        released = client.post(
            _RELEASE,
            json={"allocation_id": allocation["allocation_id"], "mode": "copy_back", "force": True},
        )
    assert released.status_code == 200, released.text
    assert (released.json()["mode"], released.json()["forced"]) == ("copy_back", True)
    assert (world.source / "plan.md").read_bytes() == b"replace a - b with a + b\n"
    assert world.sandboxes.finalizations == ["completed"]


def test_only_full_access_callers_may_list_or_release(tmp_path: Path) -> None:
    world = crashed_world(tmp_path, "run-roles")
    roles = ControlPlaneRoles()
    roles.add_api_path("/api")

    class Auth:
        control_plane_roles = roles

        def __call__(self, request):
            role = request.headers.get("x-test-role")
            if role is None:
                raise HTTPException(401, "Authentication required")
            context = AuthContext(subject="caller", claims={"role": role})
            roles.require(request, context)
            return context

    config = ServerConfig(access=AuthenticatedAccess(dependency=Auth()))
    with TestClient(create_server(world.app(), config=config)) as client:
        member = {"x-test-role": "member"}
        developer = {"x-test-role": "developer"}
        assert client.get(_LIST).status_code == 401
        assert client.get(_LIST, headers=member).status_code == 403
        (allocation,) = client.get(_LIST, headers=developer).json()["allocations"]
        body = {"allocation_id": allocation["allocation_id"], "mode": "discard", "force": True}
        assert client.post(_RELEASE, json=body).status_code == 401
        assert client.post(_RELEASE, json=body, headers=member).status_code == 403
        assert world.sandboxes.disposed == []
        released = client.post(_RELEASE, json=body, headers=developer)
    assert released.status_code == 200, released.text
    assert len(world.sandboxes.disposed) == 1


def test_the_contract_advertises_the_surface(tmp_path: Path) -> None:
    world = _World(tmp_path)
    with TestClient(create_server(world.app(), config=ServerConfig.local_development())) as client:
        surfaces = client.get("/api/contract").json()["capabilities"]["surfaces"]
    assert surfaces["shared_allocations"] == {
        "configured": True,
        "read": {"enabled": True, "unavailable_reason": None},
        "mutate": {"enabled": True, "unavailable_reason": None},
    }
