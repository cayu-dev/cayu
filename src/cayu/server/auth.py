"""Authentication primitives for the Cayu server."""

from __future__ import annotations

import base64
import inspect
import secrets
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, cast

from fastapi import Request  # noqa: TC002 - FastAPI inspects this annotation at runtime.
from pydantic import BaseModel, ConfigDict, Field, field_validator

from cayu._validation import copy_json_value, require_clean_nonblank


class AuthContext(BaseModel):
    """Authenticated caller identity and operator-action provenance.

    Authentication protects access to Cayu's server surfaces. ``tenant`` does
    not scope sessions, events, transcripts, tasks, knowledge, artifacts,
    usage, or dashboard/control-plane reads; applications must enforce any
    tenant authorization and storage isolation separately.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    subject: str = Field(description="Verified identity of the authenticated caller.")
    tenant: str | None = Field(
        default=None,
        description=(
            "Optional verified tenant identity for actor provenance only. It is not a "
            "storage partition, authorization rule, row-level filter, or tenant-isolation "
            "primitive, and it does not scope Cayu data."
        ),
    )
    claims: dict[str, Any] = Field(
        default_factory=dict,
        description="Verified authentication claims available to trusted server code.",
    )

    @field_validator("subject")
    @classmethod
    def validate_subject(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("tenant")
    @classmethod
    def validate_tenant(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("claims", mode="before")
    @classmethod
    def copy_claims(cls, value: dict[str, Any]) -> dict[str, Any]:
        copied = copy_json_value(value, "claims")
        if type(copied) is not dict:
            raise ValueError("claims must be an object.")
        return copied


AuthDependencyResult = AuthContext | Mapping[str, Any]
AuthDependency = Callable[[Any], AuthDependencyResult | Awaitable[AuthDependencyResult]]


def server_auth_dependency(auth: AuthDependency) -> Callable[[Request], Awaitable[AuthContext]]:
    """Wrap a user auth callable so routes receive a validated AuthContext."""

    async def dependency(request: Request) -> AuthContext:
        return await resolve_auth_context(auth, request)

    return dependency


async def resolve_auth_context(auth: AuthDependency, request: Request) -> AuthContext:
    """Resolve and validate a server auth dependency outside FastAPI injection."""

    result = auth(request)
    resolved: AuthDependencyResult
    if inspect.isawaitable(result):
        resolved = cast("AuthDependencyResult", await result)
    else:
        resolved = cast("AuthDependencyResult", result)
    return copy_auth_context(resolved)


def copy_auth_context(value: AuthContext | Mapping[str, Any]) -> AuthContext:
    if isinstance(value, AuthContext):
        return value.model_copy(deep=True)
    if isinstance(value, Mapping):
        return AuthContext.model_validate(dict(value))
    raise TypeError("Server auth dependencies must return AuthContext or a compatible mapping.")


class BasicAuth:
    """HTTP Basic authentication dependency for small self-hosted deployments.

    ``tenant``, when configured, is copied into ``AuthContext`` as authenticated
    actor provenance only. It does not scope or authorize access to Cayu data.
    """

    def __init__(
        self,
        *,
        username: str,
        password: str,
        realm: str = "Cayu",
        subject: str | None = None,
        tenant: str | None = None,
        claims: dict[str, Any] | None = None,
    ) -> None:
        self.username = require_clean_nonblank(username, "username")
        self.password = require_clean_nonblank(password, "password")
        self.realm = require_clean_nonblank(realm, "realm")
        self.subject = require_clean_nonblank(subject, "subject") if subject is not None else None
        self.tenant = require_clean_nonblank(tenant, "tenant") if tenant is not None else None
        copied_claims = copy_json_value(claims or {}, "claims")
        if type(copied_claims) is not dict:
            raise ValueError("claims must be an object.")
        self.claims = copied_claims

    async def __call__(self, request: Request) -> AuthContext:
        authorization = request.headers.get("authorization", "")
        scheme, separator, encoded = authorization.partition(" ")
        if separator != " " or scheme.lower() != "basic":
            raise self._unauthorized()
        try:
            decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
        except (UnicodeDecodeError, ValueError) as exc:
            raise self._unauthorized() from exc
        username, separator, password = decoded.partition(":")
        if separator != ":":
            raise self._unauthorized()

        username_matches = secrets.compare_digest(
            username.encode("utf-8"),
            self.username.encode("utf-8"),
        )
        password_matches = secrets.compare_digest(
            password.encode("utf-8"),
            self.password.encode("utf-8"),
        )
        if not (username_matches and password_matches):
            raise self._unauthorized()
        return AuthContext(
            subject=self.subject or self.username,
            tenant=self.tenant,
            claims=self.claims,
        )

    def _unauthorized(self) -> Exception:
        from fastapi import HTTPException

        return HTTPException(
            status_code=401,
            detail="Missing or invalid credentials.",
            headers={"WWW-Authenticate": f'Basic realm="{self.realm}"'},
        )
