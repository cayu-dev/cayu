"""Static dashboard serving helpers."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException


class DashboardStaticFiles(StaticFiles):
    """Serve dashboard assets with React-router fallback for page routes."""

    async def get_response(self, path: str, scope: dict[str, Any]):
        try:
            return await super().get_response(path, scope)
        except HTTPException as exc:
            if not _should_serve_index(path, scope, exc):
                raise
            return await super().get_response("index.html", scope)


def _should_serve_index(path: str, scope: dict[str, Any], exc: HTTPException) -> bool:
    if exc.status_code != 404:
        return False
    if scope.get("method") not in {"GET", "HEAD"}:
        return False
    if path == "api" or path.startswith("api/"):
        return False
    if path == "assets" or path.startswith("assets/"):
        return False
    return PurePosixPath(path).suffix == ""
