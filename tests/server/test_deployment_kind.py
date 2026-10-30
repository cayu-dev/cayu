"""Deployment labels follow explicit startup metadata, never names or access policy."""

import pytest

pytest.importorskip("fastapi")
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from cayu.applications import CayuApp
from cayu.server import AuthContext, OpenAccess, ServerConfig, create_server, mount_cayu
from cayu.server.settings import ServerSettings


@pytest.mark.parametrize("kind", ["local", "cloud", "embedded"])
@pytest.mark.parametrize("protected", [False, True])
def test_explicit_server_kind_reaches_protected_diagnostics(kind, protected):
    def authenticate(request):
        return AuthContext(subject="operator")

    config = (
        ServerConfig.protected(authenticate, deployment_kind=kind, deployment_name="cloud")
        if protected
        else ServerConfig(access=OpenAccess(), deployment_kind=kind, deployment_name="cloud")
    )
    with TestClient(create_server(CayuApp(), config=config)) as client:
        response = client.get("/api/system/diagnostics")
        assert response.status_code == 200
        assert response.json()["deployment"]["kind"] == kind
        assert (
            response.json()["capabilities"]["surfaces"]["deployment_kind"]["read"]["enabled"]
            is True
        )
        assert response.headers["cache-control"] == "private, no-store"


def test_standalone_and_mount_defaults_are_declared_by_the_entrance(monkeypatch):
    monkeypatch.setenv("CAYU_CLOUD_APPLICATION_ID", "not-kind-authority")
    with TestClient(
        create_server(CayuApp(), config=ServerConfig.local_development(deployment_name="Cloud"))
    ) as client:
        assert client.get("/api/system/diagnostics").json()["deployment"]["kind"] == "local"
    server = FastAPI()
    mount_cayu(server, CayuApp(), access=OpenAccess(), path="/nested")
    with TestClient(server) as client:
        assert (
            client.get("/nested/api/system/diagnostics").json()["deployment"]["kind"] == "embedded"
        )


def test_settings_kind_is_explicit_and_survives_serialization(monkeypatch):
    monkeypatch.setenv("CAYU_SERVER_DEPLOYMENT_KIND", "cloud")
    settings = ServerSettings(access={"mode": "open"}, _env_file=None)
    config = settings.to_config()
    restored = ServerConfig.model_validate_json(config.model_dump_json())
    assert restored.deployment_kind == "cloud"
    assert restored.safe_summary()["deployment_kind"] == "cloud"


@pytest.mark.parametrize("kind", [None, True, 1, "unknown", "Cloud", " local "])
def test_invalid_kind_cannot_reach_a_diagnostic_snapshot(kind):
    with pytest.raises(ValidationError):
        ServerConfig(access=OpenAccess(), deployment_kind=kind)
    if kind is None:  # None is the explicit absence of an optional mount declaration.
        return
    server = FastAPI()
    original_routes = list(server.routes)
    with pytest.raises((ValueError, TypeError)):
        mount_cayu(server, CayuApp(), access=OpenAccess(), deployment_kind=kind)
    assert server.routes == original_routes
