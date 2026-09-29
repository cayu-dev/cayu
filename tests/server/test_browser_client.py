from __future__ import annotations

# ruff: noqa: E402
import re
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from cayu.applications import CayuApp
from cayu.server import (
    AuthenticatedAccess,
    BasicAuth,
    OpenAccess,
    ServerApiConfig,
    ServerConfig,
    create_server,
    mount_cayu,
)
from cayu.server.contracts import (
    BROWSER_CLIENT_GUIDE_TOPIC,
    BROWSER_CLIENT_VERSION,
    SERVER_CONTRACT_VERSION,
)

_CLIENT_DIR = Path(__file__).resolve().parents[2] / "src" / "cayu" / "server" / "browser_client"


def _exported_constant(source: str, name: str) -> str:
    # Matches `export const NAME = "1"` in the module and `export declare const NAME: "1"`
    # in the declarations.
    match = re.search(rf'^export (?:declare )?const {name}(?:: | = )"([^"]*)"', source, re.M)
    assert match is not None, name
    return match.group(1)


def test_mount_cayu_serves_the_browser_client_and_its_types() -> None:
    server = FastAPI()
    mount_cayu(server, CayuApp(), access=OpenAccess())
    client = TestClient(server)

    module = client.get("/cayu/client.js")
    assert module.status_code == 200
    assert module.headers["content-type"] == "text/javascript; charset=utf-8"
    assert module.headers["cache-control"] == "no-cache"
    assert module.headers["x-content-type-options"] == "nosniff"
    assert module.content == (_CLIENT_DIR / "client.js").read_bytes()
    assert "export async function connect" in module.text

    types = client.get("/cayu/client.d.ts")
    assert types.status_code == 200
    assert types.headers["content-type"] == "application/typescript; charset=utf-8"
    assert types.content == (_CLIENT_DIR / "client.d.ts").read_bytes()

    revalidated = client.get("/cayu/client.js", headers={"If-None-Match": module.headers["etag"]})
    assert revalidated.status_code == 304
    assert revalidated.content == b""

    head = client.head("/cayu/client.js")
    assert head.status_code == 200
    assert head.content == b""
    assert head.headers["content-length"] == str(len(module.content))

    assert client.get("/cayu/", follow_redirects=False).status_code == 200
    assert "/cayu/client.js" not in client.get("/openapi.json").json()["paths"]


def test_contract_advertises_the_browser_client() -> None:
    server = FastAPI()
    mount_cayu(server, CayuApp(), path="/ops", dashboard=False, access=OpenAccess())
    client = TestClient(server)

    contract = client.get("/ops/api/contract").json()
    assert contract["client"] == {
        "module_url": "/ops/client.js",
        "types_url": "/ops/client.d.ts",
        "version": BROWSER_CLIENT_VERSION,
        "guide_topic": BROWSER_CLIENT_GUIDE_TOPIC,
    }
    assert BROWSER_CLIENT_GUIDE_TOPIC == "app-ui"
    assert client.get("/ops/client.js").status_code == 200


def test_browser_client_is_behind_the_mount_access_dependency() -> None:
    server = FastAPI()
    mount_cayu(
        server,
        CayuApp(),
        access=AuthenticatedAccess(
            dependency=BasicAuth(username="operator", password="secret-password")
        ),
    )
    client = TestClient(server)

    assert client.get("/cayu/client.js").status_code == 401
    assert client.get("/cayu/client.d.ts").status_code == 401
    accepted = client.get("/cayu/client.js", auth=("operator", "secret-password"))
    assert accepted.status_code == 200


def test_create_server_serves_the_client_next_to_its_api() -> None:
    default = TestClient(create_server(CayuApp(), config=ServerConfig.local_development()))
    assert default.get("/api/contract").json()["client"]["module_url"] == "/client.js"
    assert default.get("/client.js").status_code == 200
    assert default.get("/cayu/", follow_redirects=False).status_code == 200

    nested = TestClient(
        create_server(
            CayuApp(),
            config=ServerConfig.local_development(api=ServerApiConfig(path="/cayu/api")),
        )
    )
    assert nested.get("/cayu/api/contract").json()["client"]["types_url"] == "/cayu/client.d.ts"
    assert nested.get("/cayu/client.js").status_code == 200

    # The module finds the API at ./api, so another API layout does not serve it.
    other = TestClient(
        create_server(
            CayuApp(),
            config=ServerConfig.local_development(api=ServerApiConfig(path="/v1")),
        )
    )
    advertised = other.get("/v1/contract").json()["client"]
    assert advertised["module_url"] is None
    assert advertised["types_url"] is None
    assert other.get("/client.js").status_code == 404


def test_browser_client_is_versioned_with_the_server_contract() -> None:
    module = (_CLIENT_DIR / "client.js").read_text(encoding="utf-8")
    types = (_CLIENT_DIR / "client.d.ts").read_text(encoding="utf-8")

    assert _exported_constant(module, "CONTRACT_VERSION") == SERVER_CONTRACT_VERSION, (
        "Update CONTRACT_VERSION in client.js with the server contract version."
    )
    assert _exported_constant(module, "CLIENT_VERSION") == BROWSER_CLIENT_VERSION
    assert _exported_constant(types, "CLIENT_VERSION") == BROWSER_CLIENT_VERSION
    assert "cayu guide app-ui" in module
