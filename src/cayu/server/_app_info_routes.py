"""Read-only dashboard identity and the authenticated viewer's role."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from fastapi import APIRouter, HTTPException, Response

from cayu.server._diagnostics import _bounded_diagnostic_text
from cayu.server.auth import AuthContext
from cayu.server.contracts import MAX_SYSTEM_DEPLOYMENT_NAME_CHARS, AppInfoResponse
from cayu.server.proxy_identity import ControlPlaneRoles

if TYPE_CHECKING:
    from fastapi.params import Depends


def register_app_info_routes(
    router: APIRouter,
    *,
    project_name: str | None,
    deployment_name: str | None,
    runtime_version: str | None,
    roles: ControlPlaneRoles | None,
    optional_auth_context: Depends,
) -> None:
    """Publish bounded server identity without reading any application stores."""

    project_name, _ = _bounded_diagnostic_text(
        project_name, max_chars=MAX_SYSTEM_DEPLOYMENT_NAME_CHARS
    )
    name = project_name or deployment_name or "Cayu"

    app_auth_context = cast("AuthContext | None", optional_auth_context)

    @router.get("/app", response_model=AppInfoResponse)
    async def get_app_info(
        response: Response,
        auth_context: AuthContext | None = app_auth_context,
    ) -> AppInfoResponse:
        role = None
        full_access = True
        if roles is not None:
            value = None if auth_context is None else auth_context.claims.get(roles.role_claim)
            # The configured owner, not a default role spelling or arbitrary
            # claims on an unrelated auth dependency, determines access.
            if type(value) is not str or value not in (*roles.full_access, *roles.read_and_decide):
                raise HTTPException(status_code=403, detail={"code": "role_forbidden"})
            role = value
            full_access = role in roles.full_access
        response.headers["Cache-Control"] = "private, no-store"
        return AppInfoResponse(
            name=name,
            runtime_version=runtime_version or "unknown",
            role=role,
            full_access=full_access,
        )
