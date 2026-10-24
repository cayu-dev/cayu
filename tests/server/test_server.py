"""HTTP server assembly and dashboard mounting contracts."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")
pytest.importorskip("httpx")

import re

from fastapi import FastAPI
from fastapi.testclient import TestClient
from tests.server.server_test_support import _LOCAL_SERVER_CONFIG, OneShotProvider

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.server import (
    DashboardConfig,
    OpenAccess,
    ServerApiConfig,
    ServerConfig,
    create_server,
    mount_cayu,
    mount_dashboard,
)


def test_dashboard_routes_fall_back_to_index_without_masking_api_or_assets() -> None:
    app = CayuApp()
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    for path in ["/cayu/sessions", "/cayu/run", "/cayu/sessions/session-abc"]:
        response = client.get(path)
        assert response.status_code == 200
        assert '<div id="root"></div>' in response.text
        assert '<base href="/cayu/" />' in response.text
        assert '"basePath":"/cayu"' in response.text

    assert client.get("/sessions").status_code == 404
    assert client.get("/api/missing").status_code == 404
    assert client.get("/cayu/assets/missing.js").status_code == 404


def test_dashboard_uses_effective_paths_when_server_is_nested_under_asgi_mount() -> None:
    parent = FastAPI()
    parent.mount("/product", create_server(CayuApp(), config=_LOCAL_SERVER_CONFIG))

    client = TestClient(parent)
    response = client.get("/product/cayu/sessions/deep-link")

    assert response.status_code == 200
    assert '<base href="/product/cayu/" />' in response.text
    assert '"basePath":"/product/cayu"' in response.text
    assert '"apiBaseUrl":"/product/api"' in response.text

    asset_paths = re.findall(r'(?:src|href)="\./(assets/[^"]+)"', response.text)
    assert asset_paths
    for asset_path in asset_paths:
        assert client.get(f"/product/cayu/{asset_path}").status_code == 200
        assert client.get(f"/cayu/{asset_path}").status_code == 404


def test_dashboard_serves_lazy_route_chunks_under_nested_asgi_mount() -> None:
    parent = FastAPI()
    parent.mount("/product", create_server(CayuApp(), config=_LOCAL_SERVER_CONFIG))

    client = TestClient(parent)
    shell = client.get("/product/cayu/sessions/deep-link")

    assert shell.status_code == 200
    entry_scripts = re.findall(r'src="\./(assets/[^"]+\.js)"', shell.text)
    assert len(entry_scripts) == 1

    pending_assets = list(entry_scripts)
    visited_assets: set[str] = set()
    entry_chunks: set[str] | None = None

    while pending_assets:
        asset_path = pending_assets.pop()
        if asset_path in visited_assets:
            continue
        visited_assets.add(asset_path)

        response = client.get(f"/product/cayu/{asset_path}")
        assert response.status_code == 200, asset_path
        assert "javascript" in response.headers["content-type"], asset_path

        chunks = set(re.findall(r'["`]\./([^"`]+\.js)["`]', response.text))
        if asset_path == entry_scripts[0]:
            entry_chunks = chunks
        pending_assets.extend(f"assets/{chunk}" for chunk in chunks)

    assert entry_chunks is not None
    assert any(chunk.startswith("session-entry-") for chunk in entry_chunks)
    assert any(chunk.startswith("sessions-entry-") for chunk in entry_chunks)
    assert any(chunk.startswith("artifacts-") for chunk in entry_chunks)


def test_dashboard_path_can_be_disabled_or_customized() -> None:
    disabled = TestClient(
        create_server(
            CayuApp(),
            config=ServerConfig.local_development(dashboard=DashboardConfig(enabled=False)),
        )
    )
    assert disabled.get("/cayu/").status_code == 404

    custom = TestClient(
        create_server(
            CayuApp(),
            config=ServerConfig.local_development(dashboard=DashboardConfig(path="/inspector")),
        )
    )
    response = custom.get("/inspector/sessions")

    assert response.status_code == 200
    assert '<div id="root"></div>' in response.text
    assert '<base href="/inspector/" />' in response.text
    assert '"basePath":"/inspector"' in response.text
    assert custom.get("/cayu/").status_code == 404


def test_create_server_can_embed_api_under_dashboard_path() -> None:
    client = TestClient(
        create_server(
            CayuApp(),
            config=ServerConfig.local_development(
                api=ServerApiConfig(path="/cayu/api"),
                dashboard=DashboardConfig(path="/cayu"),
            ),
        )
    )

    dashboard = client.get("/cayu/sessions")
    assert dashboard.status_code == 200
    assert '<div id="root"></div>' in dashboard.text
    assert '<base href="/cayu/" />' in dashboard.text
    assert '"basePath":"/cayu"' in dashboard.text
    assert '"apiBaseUrl":"/cayu/api"' in dashboard.text

    assert client.get("/cayu/api/health").json() == {"ok": True}
    assert client.get("/api/health").status_code == 404
    assert client.get("/cayu/api/missing").status_code == 404


def test_mount_dashboard_helper_supports_composed_apps() -> None:
    app = FastAPI()

    assert mount_dashboard(app, dashboard_path="/inspector") is True

    client = TestClient(app)
    response = client.get("/inspector/knowledge")

    assert response.status_code == 200
    assert '<div id="root"></div>' in response.text
    assert '<base href="/inspector/" />' in response.text
    assert '"basePath":"/inspector"' in response.text


def test_mount_dashboard_owns_slashless_path_before_host_fallback() -> None:
    app = FastAPI()

    assert mount_dashboard(app, dashboard_path="/internal/agents") is True

    @app.get("/{path:path}")
    async def host_fallback(path: str) -> dict[str, str]:
        return {"app": "host", "path": path}

    client = TestClient(app)
    redirect = client.get(
        "/internal/agents?source=embed",
        follow_redirects=False,
    )

    assert redirect.status_code == 307
    assert redirect.headers["location"] == "http://testserver/internal/agents/?source=embed"

    head_redirect = client.head(
        "/internal/agents?source=probe",
        follow_redirects=False,
    )
    assert head_redirect.status_code == 307
    assert head_redirect.headers["location"] == ("http://testserver/internal/agents/?source=probe")

    dashboard = client.get(redirect.headers["location"])
    assert dashboard.status_code == 200
    assert '<div id="root"></div>' in dashboard.text
    assert '"basePath":"/internal/agents"' in dashboard.text
    assert '"apiBaseUrl":"/api"' in dashboard.text
    assert client.get("/elsewhere").json() == {"app": "host", "path": "elsewhere"}


@pytest.mark.parametrize("dashboard_path", [None, "/missing"])
def test_mount_dashboard_does_not_claim_unavailable_slashless_path(
    tmp_path, dashboard_path
) -> None:
    app = FastAPI()
    dashboard_dir = tmp_path / "missing" if dashboard_path is not None else None

    assert (
        mount_dashboard(
            app,
            dashboard_dir=dashboard_dir,
            dashboard_path=dashboard_path,
        )
        is False
    )

    @app.get("/{path:path}")
    async def host_fallback(path: str) -> dict[str, str]:
        return {"app": "host", "path": path}

    requested_path = "/missing" if dashboard_path is not None else "/cayu"
    response = TestClient(app).get(requested_path, follow_redirects=False)
    assert response.status_code == 200
    assert response.json()["app"] == "host"


def test_mount_dashboard_does_not_claim_directory_without_entrypoint(tmp_path) -> None:
    app = FastAPI()
    dashboard_dir = tmp_path / "empty-dashboard"
    dashboard_dir.mkdir()

    assert (
        mount_dashboard(
            app,
            dashboard_dir=dashboard_dir,
            dashboard_path="/inspector",
        )
        is False
    )

    @app.get("/{path:path}")
    async def host_fallback(path: str) -> dict[str, str]:
        return {"app": "host", "path": path}

    response = TestClient(app).get("/inspector", follow_redirects=False)
    assert response.status_code == 200
    assert response.json() == {"app": "host", "path": "inspector"}


def test_mount_dashboard_non_directory_does_not_claim_slashless_path(
    tmp_path,
) -> None:
    app = FastAPI()
    invalid_dashboard = tmp_path / "dashboard.html"
    invalid_dashboard.write_text("not a directory")

    routes_before = list(app.routes)
    assert (
        mount_dashboard(
            app,
            dashboard_dir=invalid_dashboard,
            dashboard_path="/inspector",
        )
        is False
    )
    assert app.routes == routes_before

    @app.get("/{path:path}")
    async def host_fallback(path: str) -> dict[str, str]:
        return {"app": "host", "path": path}

    response = TestClient(app).get("/inspector", follow_redirects=False)
    assert response.status_code == 200
    assert response.json() == {"app": "host", "path": "inspector"}


def test_root_dashboard_mount_does_not_register_redirect() -> None:
    app = FastAPI()

    assert mount_dashboard(app, dashboard_path="/") is True

    response = TestClient(app).get("/", follow_redirects=False)
    assert response.status_code == 200
    assert '"basePath":"/"' in response.text


def test_slashless_dashboard_head_redirects_to_canonical_path() -> None:
    app = FastAPI()
    assert mount_dashboard(app, dashboard_path="/inspector") is True

    response = TestClient(app).head("/inspector", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "http://testserver/inspector/"


@pytest.mark.parametrize(
    "method",
    ["POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
)
def test_slashless_dashboard_redirect_is_limited_to_get_and_head(method: str) -> None:
    app = FastAPI()
    assert mount_dashboard(app, dashboard_path="/inspector") is True

    response = TestClient(app).request(
        method,
        "/inspector",
        follow_redirects=False,
    )

    assert response.status_code == 405


def test_mount_dashboard_injects_base_before_custom_shell_assets(tmp_path) -> None:
    dashboard_dir = tmp_path / "dashboard"
    assets_dir = dashboard_dir / "assets"
    assets_dir.mkdir(parents=True)
    (dashboard_dir / "index.html").write_text(
        '<!doctype html><!-- <head data-fake=">"> -->'
        '<html><HEAD data-theme="dark" data-label="a > b">'
        '<script src="./assets/app.js"></script></HEAD><body>custom</body></html>',
        encoding="utf-8",
    )
    (assets_dir / "app.js").write_text("window.customDashboard = true", encoding="utf-8")

    app = FastAPI()
    assert (
        mount_dashboard(
            app,
            dashboard_dir=dashboard_dir,
            dashboard_path="/inspector",
        )
        is True
    )

    client = TestClient(app)
    response = client.get("/inspector/sessions/deep-link")

    assert response.status_code == 200
    assert '<base href="/inspector/" />' in response.text
    assert response.text.index("<base ") < response.text.index("./assets/app.js")
    assert client.get("/inspector/assets/app.js").status_code == 200


def test_mount_dashboard_preserves_doctype_for_custom_shell_without_head(tmp_path) -> None:
    dashboard_dir = tmp_path / "dashboard"
    assets_dir = dashboard_dir / "assets"
    assets_dir.mkdir(parents=True)
    (dashboard_dir / "index.html").write_text(
        '<!doctype html><html lang="en"><body>'
        '<script src="./assets/app.js"></script></body></html>',
        encoding="utf-8",
    )
    (assets_dir / "app.js").write_text("window.customDashboard = true", encoding="utf-8")

    app = FastAPI()
    assert mount_dashboard(app, dashboard_dir=dashboard_dir, dashboard_path="/inspector") is True

    response = TestClient(app).get("/inspector/sessions/deep-link")

    assert response.status_code == 200
    assert response.text.startswith("<!doctype html><html")
    assert '<head>\n    <base href="/inspector/" />' in response.text
    assert response.text.index("<!doctype html>") < response.text.index("<head>")
    assert response.text.index("<head>") < response.text.index("./assets/app.js")


def test_mount_cayu_mounts_api_and_dashboard_under_product_path() -> None:
    server = FastAPI()
    cayu_app = CayuApp()

    mount_cayu(server, cayu_app, path="/cayu", access=OpenAccess())

    @server.get("/{path:path}")
    async def host_fallback(path: str) -> dict[str, str]:
        return {"app": "host", "path": path}

    client = TestClient(server)
    redirect = client.get("/cayu", follow_redirects=False)
    assert redirect.status_code == 307
    assert redirect.headers["location"] == "http://testserver/cayu/"

    head_redirect = client.head("/cayu", follow_redirects=False)
    assert head_redirect.status_code == 307
    assert head_redirect.headers["location"] == "http://testserver/cayu/"

    dashboard_root = client.get(redirect.headers["location"])
    assert dashboard_root.status_code == 200
    assert '"basePath":"/cayu"' in dashboard_root.text
    assert '"apiBaseUrl":"/cayu/api"' in dashboard_root.text

    dashboard = client.get("/cayu/knowledge")

    assert dashboard.status_code == 200
    assert '<div id="root"></div>' in dashboard.text
    assert '<base href="/cayu/" />' in dashboard.text
    assert '"basePath":"/cayu"' in dashboard.text
    assert '"apiBaseUrl":"/cayu/api"' in dashboard.text
    assert client.get("/cayu/api/health").json() == {"ok": True}
    assert client.get("/api/health").json() == {"app": "host", "path": "api/health"}
    assert client.get("/elsewhere").json() == {"app": "host", "path": "elsewhere"}
    assert client.get("/cayu/api/missing").status_code == 404


def test_mount_cayu_can_disable_dashboard_for_api_only_services() -> None:
    server = FastAPI()
    cayu_app = CayuApp()

    mount_cayu(server, cayu_app, path="/cayu", dashboard=False, access=OpenAccess())

    @server.get("/{path:path}")
    async def host_fallback(path: str) -> dict[str, str]:
        return {"app": "host", "path": path}

    client = TestClient(server)
    assert client.get("/cayu/api/health").json() == {"ok": True}
    assert client.get("/cayu", follow_redirects=False).json()["app"] == "host"
