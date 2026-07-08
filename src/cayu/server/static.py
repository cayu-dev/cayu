"""Static dashboard serving helpers."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException
from starlette.responses import Response

from cayu.server.auth import AuthDependency, resolve_auth_context

if TYPE_CHECKING:
    from starlette.types import Scope

_HEAD_END = "</head>"


class DashboardStaticFiles(StaticFiles):
    """Serve dashboard assets with React-router fallback for page routes."""

    def __init__(
        self,
        *,
        directory: str,
        html: bool = True,
        auth: AuthDependency | None = None,
        base_path: str = "/",
        api_base_url: str = "/api",
        dashboard_config: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(directory=directory, html=html, **kwargs)
        self._directory = Path(directory)
        self._auth = auth
        self._base_path = _normalize_public_path(base_path)
        self._api_base_url = _normalize_public_url(api_base_url)
        self._excluded_api_path = _relative_child_path(
            base=self._base_path,
            child=self._api_base_url,
        )
        self._dashboard_config = dict(dashboard_config or {})

    async def get_response(self, path: str, scope: Scope):
        if _is_excluded_path(path, self._excluded_api_path):
            raise HTTPException(status_code=404)
        if self._auth is not None:
            await resolve_auth_context(self._auth, Request(scope))
        if _is_index_path(path, scope):
            return self._index_response(scope)
        try:
            response = await super().get_response(path, scope)
        except HTTPException as exc:
            if not _should_serve_index(path, scope, exc):
                raise
            return self._index_response(scope)
        if path == "index.html":
            return self._index_response(scope)
        return response

    def _index_response(self, scope: Scope) -> Response:
        index = self._directory / "index.html"
        if not index.exists():
            raise HTTPException(status_code=404)
        if scope.get("method") == "HEAD":
            return Response(media_type="text/html")
        text = index.read_text(encoding="utf-8")
        return Response(
            _inject_dashboard_config(
                text,
                base_path=self._base_path,
                api_base_url=self._api_base_url,
                dashboard_config=self._dashboard_config,
            ),
            media_type="text/html",
        )


def _should_serve_index(path: str, scope: Scope, exc: HTTPException) -> bool:
    if exc.status_code != 404:
        return False
    if scope.get("method") not in {"GET", "HEAD"}:
        return False
    if path == "api" or path.startswith("api/"):
        return False
    if path == "assets" or path.startswith("assets/"):
        return False
    return PurePosixPath(path).suffix == ""


def _is_index_path(path: str, scope: Scope) -> bool:
    if scope.get("method") not in {"GET", "HEAD"}:
        return False
    return path in {"", ".", "./", "index.html"}


def _inject_dashboard_config(
    html: str,
    *,
    base_path: str,
    api_base_url: str,
    dashboard_config: dict[str, Any] | None = None,
) -> str:
    config = dict(dashboard_config or {})
    config.update({"basePath": base_path, "apiBaseUrl": api_base_url})
    config_json = json.dumps(
        jsonable_encoder(config),
        separators=(",", ":"),
    ).replace("<", "\\u003c")
    script = f"<script>window.__CAYU_DASHBOARD_CONFIG__={config_json};</script>"
    if _HEAD_END in html:
        return html.replace(_HEAD_END, f"{script}{_HEAD_END}", 1)
    return f"{script}{html}"


def _normalize_public_path(path: str) -> str:
    value = path.strip()
    if not value:
        return "/"
    value = "/" + value.strip("/")
    return "/" if value == "/" else value


def _normalize_public_url(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        return "/"
    if "://" in stripped:
        return stripped.rstrip("/")
    return _normalize_public_path(stripped)


def _relative_child_path(*, base: str, child: str) -> str | None:
    if "://" in child:
        return None
    if base == "/":
        return child.strip("/") or None
    if child == base:
        return ""
    prefix = f"{base}/"
    if child.startswith(prefix):
        return child.removeprefix(prefix).strip("/") or None
    return None


def _is_excluded_path(path: str, excluded: str | None) -> bool:
    if excluded is None:
        return False
    value = path.strip("/")
    return value == excluded or value.startswith(f"{excluded}/")
