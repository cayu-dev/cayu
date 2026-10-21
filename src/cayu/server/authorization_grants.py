"""Authenticated HTTP lifecycle for durable tool authorization grants.

Mounted only when ``create_server(..., tool_authorization_grants=...)`` is
given a :class:`ToolAuthorizationGrantServer`; the router can also be included
in an application's own FastAPI app. Every route requires authentication. The
resolver actor always comes from the verified ``AuthContext`` and the call
being granted always comes from the runtime's pending-approval snapshot, so
clients cannot supply either.
"""

import base64
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator

from cayu._validation import copy_durable_json_object
from cayu.approvals.actors import ResolutionActor, ResolutionActorSource
from cayu.approvals.authorization_grant_service import (
    ToolAuthorizationGrantRequest,
    ToolAuthorizationGrantService,
)
from cayu.approvals.authorization_grants import (
    TOOL_AUTHORIZATION_GRANT_MAX_LIST,
    ToolAuthorizationGrant,
    ToolAuthorizationGrantConflict,
    ToolAuthorizationGrantDenied,
    ToolAuthorizationGrantError,
    ToolAuthorizationGrantNotFound,
    ToolAuthorizationGrantQuery,
    ToolAuthorizationGrantScopeKind,
    ToolAuthorizationGrantStatus,
    ToolAuthorizationGrantUnavailable,
    ToolAuthorizationGrantUse,
    ToolAuthorizationGrantUseCursor,
)
from cayu.approvals.tools import ToolApprovalDecision, ToolApprovalRequest
from cayu.server._inbox_access import InboxAccessGate
from cayu.server.auth import AuthContext, AuthDependency, server_auth_dependency
from cayu.server.contracts import BOUNDED_STREAMING_ENDPOINT_RESPONSES
from cayu.server.inbox import InboxAccess

GrantReadAccess = Callable[[AuthContext, ToolAuthorizationGrant], Awaitable[bool]]


class ToolAuthorizationGrantApproveBody(BaseModel):
    """Approve one pending tool call and remember it as a grant.

    There is no ``resolved_by`` and no tool, argument, or decision field: the
    actor is the authenticated caller and the call is the pending snapshot.
    """

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    session_id: str = Field(min_length=1)
    approval_id: str = Field(min_length=1)
    tool_round_id: str = Field(min_length=1)
    tool_call_id: str = Field(min_length=1)
    reason: str | None = Field(default=None, min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    grant: ToolAuthorizationGrantRequest

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: Any) -> dict[str, Any]:
        from cayu.server.routes import _copy_control_plane_metadata

        return _copy_control_plane_metadata(copy_durable_json_object(value, "metadata"))


class ToolAuthorizationGrantRevokeBody(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    reason: str | None = Field(default=None, min_length=1, max_length=2048)


class ToolAuthorizationGrantView(BaseModel):
    """A grant with its expiry-aware status at response time."""

    model_config = ConfigDict(extra="forbid")

    grant: ToolAuthorizationGrant
    effective_status: ToolAuthorizationGrantStatus

    @classmethod
    def from_grant(
        cls, grant: ToolAuthorizationGrant, *, now: datetime
    ) -> "ToolAuthorizationGrantView":
        """The public view: argument values are redacted unless they were publishable."""

        return cls(grant=grant.public_view(), effective_status=grant.effective_status(now))


class ToolAuthorizationGrantCursor(BaseModel):
    """The exclusive position to continue a listing after."""

    model_config = ConfigDict(extra="forbid")

    after_created_at: datetime
    after_grant_id: str


class ToolAuthorizationGrantListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    grants: list[ToolAuthorizationGrantView]
    #: Pass these back as query parameters to read the next page; ``None`` on the last page.
    next: ToolAuthorizationGrantCursor | None = None


class ToolAuthorizationGrantUseListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uses: list[ToolAuthorizationGrantUse]
    #: Pass back as ``cursor`` to read the next page; ``None`` on the last page.
    next_cursor: str | None = None


def _encode_use_cursor(cursor: ToolAuthorizationGrantUseCursor) -> str:
    return base64.urlsafe_b64encode(cursor.model_dump_json().encode("utf-8")).decode("ascii")


def _decode_use_cursor(value: str) -> ToolAuthorizationGrantUseCursor:
    try:
        return ToolAuthorizationGrantUseCursor.model_validate_json(
            base64.urlsafe_b64decode(value.encode("ascii"))
        )
    except (ValueError, UnicodeError):
        raise HTTPException(status_code=422, detail="Invalid use cursor.") from None


def _actor(auth: AuthContext) -> ResolutionActor:
    try:
        return ResolutionActor(
            subject=auth.subject,
            tenant=auth.tenant,
            source=ResolutionActorSource.HTTP_AUTH,
            claims=auth.claims,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _http_error(error: Exception) -> HTTPException:
    if isinstance(error, ToolAuthorizationGrantDenied):
        return HTTPException(status_code=403, detail="Grant action is not authorized.")
    if isinstance(error, ToolAuthorizationGrantNotFound | KeyError):
        return HTTPException(status_code=404, detail="Grant or approval was not found.")
    if isinstance(error, ToolAuthorizationGrantConflict | ToolAuthorizationGrantUnavailable):
        return HTTPException(status_code=409, detail=str(error))
    return HTTPException(status_code=422, detail=str(error))


class ToolAuthorizationGrantServer:
    """HTTP wrapper for a :class:`ToolAuthorizationGrantService`.

    ``authorize_read`` decides which authenticated callers may see a grant;
    create and revoke are decided by the service's grant authority.
    """

    def __init__(
        self,
        service: ToolAuthorizationGrantService,
        *,
        authorize_read: GrantReadAccess,
    ) -> None:
        if not isinstance(service, ToolAuthorizationGrantService) or not callable(authorize_read):
            raise TypeError("Grant routes require a grant service and a read access policy.")
        self.service = service
        self.authorize_read = authorize_read

    async def _view(self, principal: AuthContext, grant: ToolAuthorizationGrant):
        if await self.authorize_read(principal, grant) is not True:
            raise HTTPException(status_code=404, detail="Grant or approval was not found.")
        return ToolAuthorizationGrantView.from_grant(grant, now=self.service.now())

    def router(self, *, auth: AuthDependency, inbox_access: InboxAccess | None = None) -> APIRouter:
        """Build the grant routes.

        ``inbox_access`` is the server's inbox access rule: approving with a grant
        settles an approval, so a caller the rule hides that approval from is refused.
        """

        from cayu.server.routes import _accepted_event_stream_response

        router = APIRouter(prefix="/tool-authorization-grants", tags=["tool-authorization-grants"])
        authenticated = server_auth_dependency(auth)
        Principal = Annotated[AuthContext, Depends(authenticated)]
        service = self.service
        inbox_gate = InboxAccessGate(service.app, service.app._runtime_session_store, inbox_access)

        @router.post("/approve", responses=BOUNDED_STREAMING_ENDPOINT_RESPONSES)
        async def approve_with_grant(body: ToolAuthorizationGrantApproveBody, principal: Principal):
            """Remember and approve: idempotent by approval identity; streams the resolution."""

            if inbox_gate.applies(principal):
                try:
                    session_id = await service.app._resolve_public_session_id(body.session_id)
                    approval_id = await service.app._resolve_public_action_linkage(
                        session_id=session_id, value=body.approval_id, field_name="approval_id"
                    )
                except ValueError:
                    raise _http_error(KeyError(body.approval_id)) from None
                await inbox_gate.require_item(
                    principal, session_id, ("approval",), approval_id=approval_id
                )
            try:
                approval = ToolApprovalRequest(
                    session_id=body.session_id,
                    approval_id=body.approval_id,
                    tool_round_id=body.tool_round_id,
                    tool_call_id=body.tool_call_id,
                    decision=ToolApprovalDecision.APPROVE,
                    reason=body.reason,
                    metadata=body.metadata,
                    resolved_by=_actor(principal),
                )
                # Authorize and stage before any approval state is touched, so
                # a refusal is an ordinary HTTP error with nothing changed.
                prepared = await service.prepare(approval, body.grant)
            except (ToolAuthorizationGrantError, KeyError, ValueError) as exc:
                raise _http_error(exc) from None
            if prepared.grant.status in {
                ToolAuthorizationGrantStatus.ABANDONED,
                ToolAuthorizationGrantStatus.REVOKED,
            }:
                raise HTTPException(
                    status_code=409,
                    detail=f"The grant for this approval is {prepared.grant.status.value}.",
                )
            return await _accepted_event_stream_response(
                service.approve_and_grant(approval, body.grant),
                cayu_app=service.app,
                session_id=prepared.approval_request.session_id,
                conflict_error_types=(
                    RuntimeError,
                    TimeoutError,
                    ValueError,
                    ToolAuthorizationGrantError,
                ),
            )

        @router.get("", response_model=ToolAuthorizationGrantListResponse)
        async def list_grants(
            principal: Principal,
            policy_id: str | None = None,
            tool_name: str | None = None,
            scope_kind: ToolAuthorizationGrantScopeKind | None = None,
            scope_id: str | None = None,
            origin_session_id: str | None = None,
            status: Annotated[list[ToolAuthorizationGrantStatus] | None, Query()] = None,
            limit: Annotated[int, Query(ge=1, le=TOOL_AUTHORIZATION_GRANT_MAX_LIST)] = 100,
            after_created_at: datetime | None = None,
            after_grant_id: str | None = None,
        ):
            try:
                query = ToolAuthorizationGrantQuery(
                    policy_id=policy_id,
                    tool_name=tool_name,
                    scope_kind=scope_kind,
                    scope_id=scope_id,
                    origin_session_id=origin_session_id,
                    statuses=None if status is None else tuple(status),
                    limit=limit,
                    after_created_at=after_created_at,
                    after_grant_id=after_grant_id,
                )
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from None
            page = await service.list_grants(query)
            views = []
            for grant in page:
                if await self.authorize_read(principal, grant) is True:
                    views.append(ToolAuthorizationGrantView.from_grant(grant, now=service.now()))
            # The cursor follows the store page, so unreadable grants never end paging early.
            following = query.next_page(page)
            cursor = None if following is None else following.after()
            return ToolAuthorizationGrantListResponse(
                grants=views,
                next=(
                    None
                    if cursor is None
                    else ToolAuthorizationGrantCursor(
                        after_created_at=cursor[0], after_grant_id=cursor[1]
                    )
                ),
            )

        @router.get("/{grant_id}", response_model=ToolAuthorizationGrantView)
        async def get_grant(grant_id: str, principal: Principal):
            grant = await service.get(grant_id)
            if grant is None:
                raise HTTPException(status_code=404, detail="Grant or approval was not found.")
            return await self._view(principal, grant)

        @router.get("/{grant_id}/uses", response_model=ToolAuthorizationGrantUseListResponse)
        async def list_uses(
            grant_id: str,
            principal: Principal,
            limit: Annotated[int, Query(ge=1, le=TOOL_AUTHORIZATION_GRANT_MAX_LIST)] = 100,
            cursor: str | None = None,
        ):
            grant = await service.get(grant_id)
            if grant is None:
                raise HTTPException(status_code=404, detail="Grant or approval was not found.")
            await self._view(principal, grant)
            after = None if cursor is None else _decode_use_cursor(cursor)
            uses = await service.list_uses(grant_id, limit=limit, after=after)
            return ToolAuthorizationGrantUseListResponse(
                uses=list(uses),
                next_cursor=(
                    _encode_use_cursor(ToolAuthorizationGrantUseCursor.after(uses[-1]))
                    if len(uses) == limit
                    else None
                ),
            )

        @router.post("/{grant_id}/revoke", response_model=ToolAuthorizationGrantView)
        async def revoke_grant(
            grant_id: str, body: ToolAuthorizationGrantRevokeBody, principal: Principal
        ):
            try:
                grant = await service.revoke(grant_id, actor=_actor(principal), reason=body.reason)
            except (ToolAuthorizationGrantError, KeyError, ValueError) as exc:
                raise _http_error(exc) from None
            return await self._view(principal, grant)

        @router.post("/{grant_id}/reconcile", response_model=ToolAuthorizationGrantView)
        async def reconcile_grant(grant_id: str, principal: Principal):
            existing = await service.get(grant_id)
            if existing is None:
                raise HTTPException(status_code=404, detail="Grant or approval was not found.")
            await self._view(principal, existing)
            try:
                grant = await service.reconcile(grant_id)
            except (ToolAuthorizationGrantError, KeyError, ValueError) as exc:
                raise _http_error(exc) from None
            return await self._view(principal, grant)

        return router


__all__ = [
    "GrantReadAccess",
    "ToolAuthorizationGrantApproveBody",
    "ToolAuthorizationGrantListResponse",
    "ToolAuthorizationGrantRevokeBody",
    "ToolAuthorizationGrantServer",
    "ToolAuthorizationGrantUseListResponse",
    "ToolAuthorizationGrantView",
]
