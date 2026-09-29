"""Serve the dependency-free Cayu browser client module.

The module lives next to the control-plane API, at ``{parent}/client.js`` for an
API mounted at ``{parent}/api``, so it can find the API relative to its own URL.
It is package data and carries no deployment state, but it is served behind the
same access dependency as the API it talks to.
"""

from __future__ import annotations

from hashlib import sha256
from importlib.resources import files
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, Request
from starlette.responses import Response

from cayu.server.auth import server_auth_dependency

if TYPE_CHECKING:
    from cayu.server.auth import AuthDependency

BROWSER_CLIENT_MODULE = "client.js"
BROWSER_CLIENT_TYPES = "client.d.ts"

_MEDIA_TYPES = {
    BROWSER_CLIENT_MODULE: "text/javascript; charset=utf-8",
    BROWSER_CLIENT_TYPES: "application/typescript; charset=utf-8",
}


def browser_client_source(name: str) -> bytes:
    """Return the packaged bytes of ``client.js`` or ``client.d.ts``."""

    if name not in _MEDIA_TYPES:
        raise ValueError(f"Unknown browser client file: {name!r}.")
    return files("cayu.server").joinpath("browser_client", name).read_bytes()


def browser_client_base_path(api_path: str) -> str | None:
    """Return where the client is served for an API path, if it can be.

    The module defaults to ``./api`` relative to itself, so it is only served
    when the API path's last segment is ``api``.
    """

    parent, _, last = api_path.rstrip("/").rpartition("/")
    if last != "api":
        return None
    return parent or "/"


def browser_client_url(base_path: str, name: str) -> str:
    return f"/{name}" if base_path == "/" else f"{base_path}/{name}"


def browser_client_router(*, base_path: str, auth: AuthDependency | None) -> APIRouter:
    """Build routes for ``client.js`` and ``client.d.ts`` under ``base_path``.

    Include the router before any static mount at the same path so these
    routes take precedence. Responses carry a content ETag and
    ``Cache-Control: no-cache``, so browsers revalidate instead of running a
    client left over from an older server.
    """

    dependencies = [Depends(server_auth_dependency(auth))] if auth is not None else []
    router = APIRouter()
    for name, media_type in _MEDIA_TYPES.items():
        router.add_api_route(
            browser_client_url(base_path, name),
            _static_endpoint(browser_client_source(name), media_type),
            methods=["GET", "HEAD"],
            include_in_schema=False,
            dependencies=dependencies,
            name=f"cayu-browser-{name}",
        )
    return router


def _static_endpoint(content: bytes, media_type: str):
    etag = f'"{sha256(content).hexdigest()[:32]}"'
    headers = {
        "Cache-Control": "no-cache",
        "ETag": etag,
        "X-Content-Type-Options": "nosniff",
    }

    async def endpoint(request: Request) -> Response:
        if etag in _if_none_match(request.headers.get("if-none-match")):
            return Response(status_code=304, headers=headers)
        body = b"" if request.method == "HEAD" else content
        response = Response(body, media_type=media_type, headers=headers)
        if request.method == "HEAD":
            response.headers["Content-Length"] = str(len(content))
        return response

    return endpoint


def _if_none_match(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip().removeprefix("W/") for item in value.split(",")}
