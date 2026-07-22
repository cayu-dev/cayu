"""Static dashboard serving helpers."""

from __future__ import annotations

import json
from html import escape
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from fastapi import Request
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException
from starlette.responses import Response

from cayu.server.auth import AuthDependency, resolve_auth_context
from cayu.server.config import normalize_dashboard_runtime_config

if TYPE_CHECKING:
    from starlette.types import Scope


class _DocumentOpenParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.head_location: tuple[int, int, int] | None = None
        self.html_location: tuple[int, int, int] | None = None
        self.doctype_location: tuple[int, int, int] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in {"head", "html"}:
            return
        if tag == "head" and self.head_location is not None:
            return
        if tag == "html" and self.html_location is not None:
            return
        tag_text = self.get_starttag_text()
        if tag_text is not None:
            line, offset = self.getpos()
            location = (line, offset, len(tag_text))
            if tag == "head":
                self.head_location = location
            else:
                self.html_location = location

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_decl(self, decl: str) -> None:
        if self.doctype_location is None and decl.lstrip().lower().startswith("doctype"):
            line, offset = self.getpos()
            self.doctype_location = (line, offset, len(decl) + 3)


class DashboardStaticFiles(StaticFiles):
    """Serve dashboard assets with optional auth and React-router fallback.

    ``auth`` may accept or reject access to the shell and assets. A returned
    ``AuthContext.tenant`` is provenance only and does not filter dashboard
    assets or the data requested from ``api_base_url``.
    """

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
        self._dashboard_config = normalize_dashboard_runtime_config(
            {} if dashboard_config is None else dashboard_config,
            field_name="dashboard_config",
        )

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
        effective_base_path = _effective_base_path(self._base_path, scope)
        return Response(
            _inject_dashboard_config(
                text,
                base_path=effective_base_path,
                api_base_url=_effective_api_base_url(
                    configured_api_base_url=self._api_base_url,
                    configured_base_path=self._base_path,
                    effective_base_path=effective_base_path,
                ),
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
        config,
        separators=(",", ":"),
    ).replace("<", "\\u003c")
    script = f"<script>window.__CAYU_DASHBOARD_CONFIG__={config_json};</script>"
    base_href = "/" if base_path == "/" else f"{base_path}/"
    base = f'<base href="{escape(base_href, quote=True)}" />'
    parser = _document_open_parser(html)
    head_end = _location_end(html, parser.head_location)
    if head_end is not None:
        return f"{html[:head_end]}\n    {base}\n    {script}{html[head_end:]}"
    insertion_point = _location_end(html, parser.html_location)
    if insertion_point is None:
        insertion_point = _location_end(html, parser.doctype_location) or 0
    head = f"<head>\n    {base}\n    {script}\n  </head>"
    return f"{html[:insertion_point]}{head}{html[insertion_point:]}"


def _document_open_parser(html: str) -> _DocumentOpenParser:
    parser = _DocumentOpenParser()
    parser.feed(html)
    parser.close()
    return parser


def _location_end(html: str, location: tuple[int, int, int] | None) -> int | None:
    if location is None:
        return None
    line, offset, tag_length = location
    line_start = sum(len(part) for part in html.splitlines(keepends=True)[: line - 1])
    return line_start + offset + tag_length


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


def _effective_base_path(configured_base_path: str, scope: Scope) -> str:
    root_path = scope.get("root_path")
    if not isinstance(root_path, str) or not root_path.strip():
        return configured_base_path
    return _normalize_public_path(root_path)


def _effective_api_base_url(
    *,
    configured_api_base_url: str,
    configured_base_path: str,
    effective_base_path: str,
) -> str:
    if "://" in configured_api_base_url:
        return configured_api_base_url
    parent_path = _mount_parent_path(
        configured_base_path=configured_base_path,
        effective_base_path=effective_base_path,
    )
    if parent_path == "/":
        return configured_api_base_url
    if configured_api_base_url == "/":
        return parent_path
    return f"{parent_path}{configured_api_base_url}"


def _mount_parent_path(*, configured_base_path: str, effective_base_path: str) -> str:
    if configured_base_path == "/":
        return effective_base_path
    if effective_base_path == configured_base_path:
        return "/"
    if effective_base_path.endswith(configured_base_path):
        parent = effective_base_path[: -len(configured_base_path)]
        return _normalize_public_path(parent)
    return "/"


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
