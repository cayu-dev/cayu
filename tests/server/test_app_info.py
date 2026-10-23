from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")
pytest.importorskip("jwt")

from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import FastAPI
from fastapi.testclient import TestClient
from tests.server.test_proxy_identity import FakeJwks, _as, _auth, _token

from cayu.applications import CayuApp
from cayu.project_control_plane import (
    ProjectControlPlaneAccess,
    _create_project_control_plane_context,
)
from cayu.server import (
    AuthContext,
    AuthenticatedAccess,
    BasicAuth,
    ControlPlaneRoles,
    ServerConfig,
    create_router,
    create_server,
    mount_cayu,
)


def test_app_info_without_role_enforcement_ignores_role_claims_and_headers():
    def auth(_request):
        return AuthContext(subject="reader", claims={"role": "member"})

    for config in (ServerConfig.local_development(), ServerConfig.protected(auth)):
        client = TestClient(create_server(CayuApp(), config=config))
        response = client.get("/api/app", headers={"X-Cayu-Identity": "not-authenticated"})
        assert response.status_code == 200
        body = response.json()
        assert set(body) == {"name", "runtime_version", "role", "full_access"}
        assert isinstance(body["name"], str)
        assert (
            body["runtime_version"]
            == client.get("/api/contract").json()["capabilities"]["cayu_version"]
        )
        assert body["role"] is None
        assert body["full_access"] is True
        assert response.headers["cache-control"] == "private, no-store"


@pytest.mark.parametrize(
    "role,full_access", [("admin", True), ("developer", True), ("member", False)]
)
def test_app_info_reports_verified_proxy_role(role, full_access):
    key = ec.generate_private_key(ec.SECP256R1())
    client = TestClient(
        create_server(CayuApp(), config=ServerConfig.protected(_auth(FakeJwks(key))))
    )
    response = client.get("/api/app", headers=_as(_token(key, role=role)))
    assert response.status_code == 200
    assert response.json()["role"] == role
    assert response.json()["full_access"] is full_access


def test_app_info_fallback_is_admin_and_invalid_proxy_does_not_fall_back():
    key = ec.generate_private_key(ec.SECP256R1())
    auth = _auth(FakeJwks(key), fallback=BasicAuth(username="reader", password="fixture-only"))
    client = TestClient(create_server(CayuApp(), config=ServerConfig.protected(auth)))
    assert client.get("/api/app").status_code == 401
    response = client.get("/api/app", auth=("reader", "fixture-only"))
    assert response.status_code == 200
    assert response.json()["role"] == "admin"
    assert response.json()["full_access"] is True
    refused = client.get("/api/app", auth=("reader", "fixture-only"), headers=_as("invalid"))
    assert refused.status_code == 401
    assert "runtime_version" not in refused.json()


def test_app_info_uses_configured_role_policy_and_never_caches_another_viewer():
    key = ec.generate_private_key(ec.SECP256R1())
    rules = ControlPlaneRoles(full_access=("admin",), read_and_decide=("developer", "member"))
    auth = _auth(FakeJwks(key), control_plane_roles=rules)
    parent = FastAPI()
    parent.mount("/product", create_server(CayuApp(), config=ServerConfig.protected(auth)))
    client = TestClient(parent)
    for role, full_access in [
        ("admin", True),
        ("developer", False),
        ("member", False),
        ("admin", True),
    ]:
        response = client.get("/product/api/app", headers=_as(_token(key, role=role)))
        assert response.status_code == 200
        assert response.json()["role"] == role
        assert response.json()["full_access"] is full_access
        assert response.headers["cache-control"] == "private, no-store"


def test_app_info_capability_is_always_a_read_only_surface():
    client = TestClient(create_server(CayuApp(), config=ServerConfig.local_development()))
    body = client.get("/api/contract").json()
    assert body["capabilities"]["surfaces"]["app_info"] == {
        "configured": True,
        "read": {"enabled": True, "unavailable_reason": None},
        "mutate": {"enabled": False, "unavailable_reason": "unsupported"},
    }


def test_app_info_openapi_has_a_typed_body_and_no_caller_identity_parameters():
    client = TestClient(create_server(CayuApp(), config=ServerConfig.local_development()))
    schema = client.get("/openapi.json").json()
    operation = schema["paths"]["/api/app"]["get"]
    assert operation.get("parameters", []) == []
    assert operation["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/AppInfoResponse"
    }
    assert set(schema["components"]["schemas"]["AppInfoResponse"]["required"]) == {
        "name",
        "runtime_version",
        "role",
        "full_access",
    }


@pytest.mark.parametrize("entrance", ["mount", "router"])
def test_app_info_uses_embedded_prefix_and_generic_name(entrance):
    key = ec.generate_private_key(ec.SECP256R1())
    auth = _auth(FakeJwks(key))
    server = FastAPI()
    if entrance == "mount":
        mount_cayu(server, CayuApp(), path="/nested", access=AuthenticatedAccess(dependency=auth))
    else:
        auth.control_plane_roles.add_api_path("/nested/api")
        app = CayuApp()
        server.include_router(
            create_router(
                cayu_app=app,
                session_store=app.session_store,
                task_store=None,
                api_path="/nested/api",
                auth=auth,
            )
        )
    response = TestClient(server).get("/nested/api/app", headers=_as(_token(key, role="member")))
    assert response.status_code == 200
    assert response.json()["name"] == "Cayu"
    assert response.json()["role"] == "member"
    assert response.json()["full_access"] is False


@pytest.mark.parametrize(
    "project_name,expected",
    [("Accounting", "Accounting"), (None, "staging"), ("x" * 129, "staging")],
)
def test_app_info_name_uses_bounded_project_then_deployment(tmp_path, project_name, expected):
    context = _create_project_control_plane_context(
        project_root=tmp_path,
        project_id=project_name,
        eval_store=None,
        store_backend=None,
        store_source=None,
        access=ProjectControlPlaneAccess.TRUSTED_LOCAL_DEVELOPMENT,
    )
    server = create_server(
        CayuApp(),
        config=ServerConfig.local_development(deployment_name="staging"),
        project_context=context,
    )
    response = TestClient(server).get("/api/app")
    assert response.status_code == 200
    assert response.json()["name"] == expected


def test_app_info_does_not_render_omitted_deployment_name():
    server = create_server(
        CayuApp(), config=ServerConfig.local_development(deployment_name="x" * 129)
    )
    assert TestClient(server).get("/api/app").json()["name"] == "Cayu"


@pytest.mark.parametrize("role,full_access", [("owner", True), ("auditor", False)])
def test_app_info_preserves_custom_proxy_roles(role, full_access):
    key = ec.generate_private_key(ec.SECP256R1())
    auth = _auth(
        FakeJwks(key),
        roles=("owner", "auditor"),
        control_plane_roles=ControlPlaneRoles(full_access=("owner",), read_and_decide=("auditor",)),
    )
    response = TestClient(create_server(CayuApp(), config=ServerConfig.protected(auth))).get(
        "/api/app", headers=_as(_token(key, role=role))
    )
    assert response.status_code == 200
    assert response.json()["role"] == role
    assert response.json()["full_access"] is full_access


def test_app_info_uses_custom_rule_claim_and_resolves_auth_once():
    class Auth:
        control_plane_roles = ControlPlaneRoles(
            role_claim="access", full_access=("owner",), read_and_decide=("reader",)
        )
        calls = 0

        def __call__(self, request):
            self.calls += 1
            context = AuthContext(subject="reader", claims={"role": "admin", "access": "reader"})
            self.control_plane_roles.require(request, context)
            return context

    auth = Auth()
    response = TestClient(create_server(CayuApp(), config=ServerConfig.protected(auth))).get(
        "/api/app"
    )
    assert response.status_code == 200
    assert response.json()["role"] == "reader"
    assert response.json()["full_access"] is False
    assert auth.calls == 1


@pytest.mark.parametrize("claims", [{"remove": ("role",)}, {"role": True}, {"role": "future-role"}])
def test_app_info_refuses_missing_malformed_or_unknown_proxy_roles(claims):
    key = ec.generate_private_key(ec.SECP256R1())
    client = TestClient(
        create_server(CayuApp(), config=ServerConfig.protected(_auth(FakeJwks(key))))
    )
    assert client.get("/api/app", headers=_as(_token(key, role="member"))).status_code == 200
    response = client.get("/api/app", headers=_as(_token(key, **claims)))
    assert response.status_code == 401
    assert "runtime_version" not in response.json()
    assert (
        client.get("/api/app", headers=_as(_token(key, role="admin"))).json()["full_access"] is True
    )
