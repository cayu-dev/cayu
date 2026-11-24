"""Authenticated HTTP routes to issue, read and revoke tool-call preauthorizations.

Mounted only when ``create_server(..., tool_call_preauthorizations=...)`` is
given a :class:`ToolCallPreauthorizationServer` (``cayu serve`` does this for
an application that calls ``app.host_tool_call_preauthorizations``). Every
route requires authentication, and the person a record names as
``authorized_by`` (or ``revoked_by``) is always the verified caller, never a
field of the request.
"""

import secrets
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from cayu._validation import copy_durable_json_object
from cayu.approvals.actors import ResolutionActorSource
from cayu.approvals.authorization_grants import (
    TOOL_AUTHORIZATION_GRANT_MAX_LIFETIME_SECONDS,
    ToolAuthorizationGrantActor,
    ToolAuthorizationGrantConflict,
    ToolAuthorizationGrantNotFound,
)
from cayu.approvals.tool_call_preauthorization_hosting import (
    ToolCallPreauthorizationAccess,
    ToolCallPreauthorizationAction,
)
from cayu.approvals.tool_call_preauthorizations import (
    ToolCallPreauthorization,
    ToolCallPreauthorizationStore,
)
from cayu.server.auth import AuthContext, AuthDependency, server_auth_dependency

_NOT_FOUND = "Preauthorization was not found."
_FORBIDDEN = "This preauthorization action is not authorized."


class ToolCallPreauthorizationIssueBody(BaseModel):
    """One exact call to preauthorize for a task.

    There is no ``authorized_by``: the record names the authenticated caller.
    ``arguments`` are hashed and not stored. ``preauthorization_id`` makes a retry
    idempotent; without one the server picks a new id.
    """

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    task_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any]
    agent_name: str | None = Field(default=None, min_length=1)
    preauthorization_id: str | None = Field(default=None, min_length=1)
    lifetime_seconds: StrictInt | None = Field(
        default=None, ge=1, le=TOOL_AUTHORIZATION_GRANT_MAX_LIFETIME_SECONDS
    )
    justification: str | None = Field(default=None, min_length=1)

    @field_validator("arguments", mode="before")
    @classmethod
    def copy_arguments(cls, value: Any) -> dict[str, Any]:
        return copy_durable_json_object(value, "arguments")


class ToolCallPreauthorizationView(BaseModel):
    """A record with what it can do now."""

    model_config = ConfigDict(extra="forbid")

    preauthorization: ToolCallPreauthorization
    status: Literal["available", "used", "revoked", "expired"]

    @classmethod
    def of(
        cls, record: ToolCallPreauthorization, *, now: datetime
    ) -> "ToolCallPreauthorizationView":
        if record.use is not None:
            status = "used"
        elif record.revoked_at is not None:
            status = "revoked"
        elif record.available(now):
            status = "available"
        else:
            status = "expired"
        return cls(preauthorization=record, status=status)


class ToolCallPreauthorizationListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preauthorizations: list[ToolCallPreauthorizationView]


def _actor(auth: AuthContext) -> ToolAuthorizationGrantActor:
    try:
        return ToolAuthorizationGrantActor(
            subject=auth.subject, tenant=auth.tenant, source=ResolutionActorSource.HTTP_AUTH
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class ToolCallPreauthorizationServer:
    """HTTP wrapper for a :class:`~cayu.approvals.ToolCallPreauthorizationStore`.

    ``authorize`` decides every issue, read and revoke for the authenticated caller.
    """

    def __init__(
        self,
        store: ToolCallPreauthorizationStore,
        *,
        authorize: ToolCallPreauthorizationAccess,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(store, ToolCallPreauthorizationStore) or not callable(authorize):
            raise TypeError(
                "Preauthorization routes require a preauthorization store and an access policy."
            )
        self.store = store
        self.authorize = authorize
        self._clock = clock or (lambda: datetime.now(UTC))

    async def _require(
        self,
        principal: AuthContext,
        action: ToolCallPreauthorizationAction,
        record: ToolCallPreauthorization,
    ) -> None:
        if await self.authorize(principal, action, record) is not True:
            if action == "read":
                raise HTTPException(status_code=404, detail=_NOT_FOUND)
            raise HTTPException(status_code=403, detail=_FORBIDDEN)

    async def _readable(self, principal: AuthContext, preauthorization_id: str):
        try:
            record = await self.store.get(preauthorization_id)
        except ToolAuthorizationGrantNotFound:
            raise HTTPException(status_code=404, detail=_NOT_FOUND) from None
        await self._require(principal, "read", record)
        return record

    def router(self, *, auth: AuthDependency) -> APIRouter:
        router = APIRouter(
            prefix="/tool-call-preauthorizations", tags=["tool-call-preauthorizations"]
        )
        Principal = Annotated[AuthContext, Depends(server_auth_dependency(auth))]

        @router.post("", status_code=201, response_model=ToolCallPreauthorizationView)
        async def issue(body: ToolCallPreauthorizationIssueBody, principal: Principal):
            """Preauthorize one exact call for a task, naming the caller as its author."""

            try:
                record = ToolCallPreauthorization.for_call(
                    preauthorization_id=body.preauthorization_id or f"tcp_{secrets.token_hex(16)}",
                    task_id=body.task_id,
                    tool_name=body.tool_name,
                    arguments=body.arguments,
                    agent_name=body.agent_name,
                    authorized_by=_actor(principal),
                    created_at=self._clock(),
                    lifetime_seconds=body.lifetime_seconds,
                    justification=body.justification,
                )
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from None
            await self._require(principal, "issue", record)
            # A retry, even one racing the first request, is the same issue: the store
            # compares terms without the issue time and keeps the first record.
            try:
                stored = await self.store.issue(record)
            except ToolAuthorizationGrantConflict as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from None
            return ToolCallPreauthorizationView.of(stored, now=self._clock())

        @router.get("", response_model=ToolCallPreauthorizationListResponse)
        async def list_for_task(task_id: str, principal: Principal):
            """The task's records the caller may read, oldest first."""

            views = []
            for record in await self.store.list_for_task(task_id):
                if await self.authorize(principal, "read", record) is True:
                    views.append(ToolCallPreauthorizationView.of(record, now=self._clock()))
            return ToolCallPreauthorizationListResponse(preauthorizations=views)

        @router.get("/{preauthorization_id:path}", response_model=ToolCallPreauthorizationView)
        async def show(preauthorization_id: str, principal: Principal):
            record = await self._readable(principal, preauthorization_id)
            return ToolCallPreauthorizationView.of(record, now=self._clock())

        @router.post(
            "/{preauthorization_id:path}/revoke", response_model=ToolCallPreauthorizationView
        )
        async def revoke(preauthorization_id: str, principal: Principal):
            """Revoke an unused record; a used one answers 409."""

            record = await self._readable(principal, preauthorization_id)
            await self._require(principal, "revoke", record)
            try:
                revoked = await self.store.revoke(
                    preauthorization_id, revoked_by=_actor(principal), now=self._clock()
                )
            except ToolAuthorizationGrantConflict as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from None
            return ToolCallPreauthorizationView.of(revoked, now=self._clock())

        return router


__all__ = [
    "ToolCallPreauthorizationIssueBody",
    "ToolCallPreauthorizationListResponse",
    "ToolCallPreauthorizationServer",
    "ToolCallPreauthorizationView",
]
