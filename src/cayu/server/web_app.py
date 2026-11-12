"""Serve a project's built web app at ``/``, next to Cayu's routes and the project's own.

``serve_web_app(server, directory)`` answers every request that no route of ``server``
answers, so Cayu's routes and the project's own always come first:

- a ``GET`` or ``HEAD`` of a file in ``directory`` answers with that file. Files under
  ``assets/`` are named by their content, so they are cached for a year; every other file,
  ``index.html`` included, is checked again on each use;
- a ``GET`` or ``HEAD`` of any other path answers with ``index.html``, so an app that routes
  in the browser opens on a reload or a shared link (``/conversations/ID``);
- a missing file under ``assets/``, any path under a reserved prefix, and every other
  method answer 404, exactly as they would without the web app.

A reserved prefix is the first segment of every route ``server`` has when a request comes
in (``/api``, ``/cayu``, ``/client.js``, the project's own such as ``/app``), plus
:data:`RESERVED_PREFIXES` and the ``reserved`` the caller adds. So a mistyped ``/api/...``
or ``/app/...`` is still a 404, never the page.

``cayu serve`` calls it for ``[tool.cayu.serve] ui = "DIRECTORY"``, after the project's
routers. An application that builds its own server with ``create_server`` or
``mount_cayu`` can call it the same way, last.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from cayu._web_app_files import (
    ASSETS,
    PAGE,
    RESERVED_PREFIXES,
    WebAppError,
    check_web_app_directory,
)

if TYPE_CHECKING:
    from starlette.responses import FileResponse
    from starlette.types import ASGIApp, Receive, Scope, Send

__all__ = [
    "RESERVED_PREFIXES",
    "WebAppError",
    "check_web_app_directory",
    "reserved_prefixes",
    "serve_web_app",
]

# The page is checked again on each use: it names the current build's files.
_PAGE_HEADERS = {
    "Cache-Control": "no-cache",
    # Never inside another site's frame, where its buttons could be clicked for the
    # signed-in person. The page can narrow its own policy further with a <meta> tag.
    "Content-Security-Policy": "frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff",
}
_ASSET_HEADERS = {
    "Cache-Control": "public, max-age=31536000, immutable",
    "X-Content-Type-Options": "nosniff",
}


def reserved_prefixes(server: Any, extra: Iterable[str] = ()) -> frozenset[str]:
    """The first segment of each of ``server``'s routes (``/api``, ``/app``), Cayu's own
    reserved prefixes, and ``extra``."""

    found = {*RESERVED_PREFIXES, *extra}
    for route in getattr(server, "routes", ()):
        path = getattr(route, "path", "")
        if path.startswith("/") and path != "/":
            found.add("/" + path.split("/")[1])
    return frozenset(found)


def serve_web_app(
    server: Any, directory: str | os.PathLike[str], *, reserved: Iterable[str] = ()
) -> None:
    """Serve the built web app in ``directory`` at ``/`` for every request no route of
    ``server`` (a FastAPI or Starlette app) answers. Call it after every other route is in
    place. Raises :class:`WebAppError` when the directory can't be served."""

    extra = tuple(reserved)
    root = check_web_app_directory(directory, reserved_prefixes(server, extra))
    router = server.router
    router.default = _WebAppFallback(root, server, extra, router.default)


class _WebAppFallback:
    """The router's answer when no route matches: a file of the app, or its page."""

    def __init__(self, root: Path, server: Any, reserved: tuple[str, ...], fallback: ASGIApp):
        self._root = root
        self._server = server
        self._reserved = reserved
        self._fallback = fallback

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        response = self._response(scope)
        if response is None:
            await self._fallback(scope, receive, send)
        else:
            await response(scope, receive, send)

    def _response(self, scope: Scope) -> FileResponse | None:
        from starlette.responses import FileResponse

        if scope["type"] != "http" or scope["method"] not in ("GET", "HEAD"):
            return None
        parts = PurePosixPath(_route_path(scope)).parts[1:]
        if parts and f"/{parts[0]}" in reserved_prefixes(self._server, self._reserved):
            return None
        file = self._file(parts)
        if file is not None:
            headers = _ASSET_HEADERS if parts[0] == ASSETS else _PAGE_HEADERS
            return FileResponse(file, headers=headers)
        if parts and parts[0] == ASSETS:
            return None
        return FileResponse(self._root / PAGE, headers=_PAGE_HEADERS)

    def _file(self, parts: tuple[str, ...]) -> Path | None:
        # No hidden files, nothing outside the directory, and no name the filesystem
        # refuses (a NUL byte, say), which is simply not a file of the app.
        if not parts or any(part.startswith(".") or "\x00" in part for part in parts):
            return None
        try:
            candidate = self._root.joinpath(*parts).resolve()
            if not candidate.is_relative_to(self._root) or not candidate.is_file():
                return None
        except (OSError, ValueError):
            return None
        return candidate


def _route_path(scope: Scope) -> str:
    """The request's path within this app: without the prefix it is mounted under
    (``root_path``), as Starlette's own router and ``StaticFiles`` read it."""

    path: str = scope["path"]
    root: str = scope.get("root_path", "")
    if root and path.startswith(root) and (len(path) == len(root) or path[len(root)] == "/"):
        return path[len(root) :] or "/"
    return path
