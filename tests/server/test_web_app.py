"""`serve_web_app`: a project's built web app at `/`, behind Cayu's routes and the project's."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from starlette.applications import Starlette
from starlette.routing import Mount

from cayu import CayuApp
from cayu.server import ServerConfig, WebAppError, create_server, serve_web_app

PAGE = '<!doctype html><script type="module" src="/assets/index-abc123.js"></script>'


def _build(root: Path) -> Path:
    static = root / "static"
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text(PAGE, encoding="utf-8")
    (static / "assets" / "index-abc123.js").write_text("console.log(1)", encoding="utf-8")
    (static / "THIRD_PARTY_LICENSES.md").write_text("# Licenses", encoding="utf-8")
    (static / ".env").write_text("SECRET=1", encoding="utf-8")
    return static


def _server(static: Path | None) -> FastAPI:
    server = create_server(CayuApp(enable_logging=False), config=ServerConfig.local_development())
    routes = APIRouter(prefix="/app")

    @routes.get("/hello")
    def hello() -> dict[str, str]:
        return {"hello": "world"}

    @routes.post("/things")
    def things() -> dict[str, str]:
        return {"made": "thing"}

    server.include_router(routes)
    if static is not None:
        serve_web_app(server, static)
    return server


def test_every_page_of_the_app_answers_with_its_page(tmp_path: Path) -> None:
    with TestClient(_server(_build(tmp_path))) as client:
        for path in ("/", "/conversations/some-id", "/settings", "/conversations/", "/a/b/c"):
            page = client.get(path)
            assert page.status_code == 200, path
            assert page.text == PAGE
            assert page.headers["cache-control"] == "no-cache"
            assert page.headers["content-security-policy"] == "frame-ancestors 'none'"
        assert client.head("/members").status_code == 200


def test_built_files_are_served_and_a_missing_asset_is_a_404(tmp_path: Path) -> None:
    with TestClient(_server(_build(tmp_path))) as client:
        asset = client.get("/assets/index-abc123.js")
        assert asset.status_code == 200 and asset.text == "console.log(1)"
        assert "immutable" in asset.headers["cache-control"]
        licenses = client.get("/THIRD_PARTY_LICENSES.md")
        assert licenses.status_code == 200 and licenses.headers["cache-control"] == "no-cache"
        for missing in ("/assets/missing.js", "/assets/nested/missing.js", "/assets"):
            assert client.get(missing).status_code == 404, missing
        # Hidden files and anything outside the directory are never files of the app.
        assert client.get("/.env").text == PAGE
        assert client.get("/%2e%2e/static/index.html").text == PAGE


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/api/health"),
        ("GET", "/api/no-such-route"),
        ("GET", "/api/sessions/"),
        ("POST", "/api/no-such-route"),
        ("GET", "/cayu/"),
        ("GET", "/cayu/sessions/some-id"),
        ("GET", "/.cayu/sign-in"),
        ("GET", "/client.js"),
        ("GET", "/app/hello"),
        ("GET", "/app/no-such-route"),
        ("GET", "/app/things"),
        ("POST", "/app/things"),
        ("POST", "/conversations"),
        ("DELETE", "/settings"),
    ],
)
def test_every_other_route_answers_as_it_did_without_the_web_app(
    tmp_path: Path, method: str, path: str
) -> None:
    static = _build(tmp_path)
    with TestClient(_server(None)) as before, TestClient(_server(static)) as after:
        expected = before.request(method, path, follow_redirects=False)
        actual = after.request(method, path, follow_redirects=False)
    assert (actual.status_code, actual.content) == (expected.status_code, expected.content)


def test_a_directory_that_cant_be_served_is_refused(tmp_path: Path) -> None:
    with pytest.raises(WebAppError, match="doesn't exist"):
        _server(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(WebAppError, match="has no index.html"):
        _server(empty)
    static = _build(tmp_path)
    (static / "app").mkdir()
    (static / "api.json").write_text("{}", encoding="utf-8")
    with pytest.raises(WebAppError, match="has /app, which the server's own routes answer"):
        _server(static)


def test_an_app_mounted_under_a_prefix_serves_its_paths_within_it(tmp_path: Path) -> None:
    host = Starlette(routes=[Mount("/product", app=_server(_build(tmp_path)))])
    with TestClient(host) as client:
        assert client.get("/product/assets/index-abc123.js").text == "console.log(1)"
        assert client.get("/product/assets/missing.js").status_code == 404
        assert client.get("/product/api/no-such-route").status_code == 404
        assert client.get("/product/app/hello").json() == {"hello": "world"}
        assert client.get("/product/conversations/some-id").text == PAGE


def test_a_path_the_filesystem_refuses_is_never_a_server_error(tmp_path: Path) -> None:
    with TestClient(_server(_build(tmp_path)), raise_server_exceptions=False) as client:
        assert client.get("/conversations/a%00b").text == PAGE
        assert client.get("/assets/x%00.js").status_code == 404
        assert client.get("/%00").status_code == 200
