"""Create, settle, list, and revoke grants from the runtime's pending approvals."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Callable, Iterable
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from cayu._clock import utc_clock
from cayu._validation import copy_durable_json_value, copy_durable_metadata
from cayu.approvals.authorization_grants import (
    TOOL_AUTHORIZATION_GRANT_MAX_LIFETIME_SECONDS,
    TOOL_AUTHORIZATION_GRANT_MAX_LIST,
    TOOL_AUTHORIZATION_GRANT_MAX_PINNED_ARGUMENTS,
    TOOL_AUTHORIZATION_GRANT_METADATA_KEY,
    TOOL_AUTHORIZATION_GRANT_OFFER_METADATA_KEY,
    TOOL_AUTHORIZATION_GRANT_SAMPLE_BUCKETS,
    ToolAuthorizationGrant,
    ToolAuthorizationGrantActor,
    ToolAuthorizationGrantAuthority,
    ToolAuthorizationGrantConflict,
    ToolAuthorizationGrantCreation,
    ToolAuthorizationGrantDenied,
    ToolAuthorizationGrantNotFound,
    ToolAuthorizationGrantOffer,
    ToolAuthorizationGrantOrigin,
    ToolAuthorizationGrantQuery,
    ToolAuthorizationGrantScope,
    ToolAuthorizationGrantScopeKind,
    ToolAuthorizationGrantStatus,
    ToolAuthorizationGrantStore,
    ToolAuthorizationGrantUnavailable,
    ToolAuthorizationGrantUse,
    ToolAuthorizationGrantUseCursor,
    ToolCallGrantMatcher,
    grant_matcher_from_arguments,
    normalize_grant_timestamp,
    tool_authorization_grant_id,
)
from cayu.approvals.tools import (
    PendingToolApproval,
    ToolApprovalDecision,
    ToolApprovalRequest,
    copy_tool_approval_request,
)
from cayu.events import Event, EventType
from cayu.sessions.event_queries import EventQuery

if TYPE_CHECKING:
    from cayu.applications import CayuApp
    from cayu.approvals.actors import ResolutionActor
    from cayu.collaboration import CollaborationAccessContext


#: Classifications retried when a grant changes while it is being reconciled.
_RECONCILE_ATTEMPTS = 3


class ToolAuthorizationGrantRequest(BaseModel):
    """The reusable authority a resolver asks to remember with one approval.

    Everything about the call itself (tool, gate, arguments, boundary) comes
    from the runtime's pending snapshot. ``pinned_argument_paths`` only names
    which snapshot values a class-of-decision grant keeps fixed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    scope: ToolAuthorizationGrantScopeKind
    expires_in_seconds: StrictInt | None = Field(
        default=None, ge=1, le=TOOL_AUTHORIZATION_GRANT_MAX_LIFETIME_SECONDS
    )
    # ``None`` uses the service's ``default_sampled_review_rate_bps``.
    sampled_review_rate_bps: StrictInt | None = Field(
        default=None, ge=0, le=TOOL_AUTHORIZATION_GRANT_SAMPLE_BUCKETS
    )
    pinned_argument_paths: tuple[str, ...] = Field(
        default=(), max_length=TOOL_AUTHORIZATION_GRANT_MAX_PINNED_ARGUMENTS
    )
    justification: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)

    @field_validator("pinned_argument_paths")
    @classmethod
    def validate_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("pinned_argument_paths must be unique.")
        return value

    @field_validator("evidence", mode="before")
    @classmethod
    def copy_evidence(cls, value: Any) -> Any:
        return copy_durable_json_value(value, "evidence")


class PreparedToolAuthorizationGrant(BaseModel):
    """A staged grant and the exact approval request that activates it."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    grant: ToolAuthorizationGrant
    approval_request: ToolApprovalRequest


class ToolAuthorizationGrantService:
    """Runtime-boundary lifecycle for grants backed by one ``CayuApp``.

    Creation reads the pending approval from the session checkpoint, checks the
    grant offer the policy recorded there, and asks ``authority`` before writing
    anything. The grant is written STAGED, keyed by the approval identity, and
    the approval stays pending. It becomes ACTIVE only when the approval is
    durably approved by a request that carried this grant's marker; an
    approval that is denied, expires, or is approved without the marker leaves
    it ABANDONED. A crash between any two steps therefore either leaves the
    approval pending with an inert grant, or leaves an approved call whose
    grant activates on the next :meth:`reconcile` or retry.
    """

    def __init__(
        self,
        app: CayuApp,
        *,
        store: ToolAuthorizationGrantStore,
        authority: ToolAuthorizationGrantAuthority,
        default_sampled_review_rate_bps: int = 0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(store, ToolAuthorizationGrantStore):
            raise TypeError("store must be a ToolAuthorizationGrantStore.")
        if not isinstance(authority, ToolAuthorizationGrantAuthority):
            raise TypeError("authority must be a ToolAuthorizationGrantAuthority.")
        self._app = app
        self._store = store
        self._authority = authority
        if (
            type(default_sampled_review_rate_bps) is not int
            or not 0 <= default_sampled_review_rate_bps <= TOOL_AUTHORIZATION_GRANT_SAMPLE_BUCKETS
        ):
            raise ValueError("default_sampled_review_rate_bps must be an integer from 0 to 10000.")
        self._default_sampled_review_rate_bps = default_sampled_review_rate_bps
        self._clock = utc_clock(clock)

    @property
    def app(self) -> CayuApp:
        return self._app

    @property
    def store(self) -> ToolAuthorizationGrantStore:
        return self._store

    def now(self) -> datetime:
        """The service clock used for creation, expiry, and settlement times."""

        return self._clock()

    @property
    def default_sampled_review_rate_bps(self) -> int:
        return self._default_sampled_review_rate_bps

    def _rate(self, grant: ToolAuthorizationGrantRequest) -> int:
        if grant.sampled_review_rate_bps is None:
            return self._default_sampled_review_rate_bps
        return grant.sampled_review_rate_bps

    async def prepare(
        self,
        approval: ToolApprovalRequest,
        grant: ToolAuthorizationGrantRequest,
        *,
        context: CollaborationAccessContext | None = None,
    ) -> PreparedToolAuthorizationGrant:
        """Stage a grant for a pending approval without resolving the approval.

        The caller submits ``approval_request`` through any public resolution
        path, then calls :meth:`reconcile`. Calling ``prepare`` again for the
        same approval and terms returns the same grant.
        """

        if type(approval) is not ToolApprovalRequest:
            raise TypeError("approval must be a ToolApprovalRequest.")
        if type(grant) is not ToolAuthorizationGrantRequest:
            raise TypeError("grant must be a ToolAuthorizationGrantRequest.")
        if approval.decision is not ToolApprovalDecision.APPROVE:
            raise ValueError("A grant can only be created with an approval.")
        if approval.task_worker_id is not None or approval.task_handoff_id is not None:
            raise ValueError(
                "A task continuation relays an approval; it cannot create a standing grant."
            )
        if TOOL_AUTHORIZATION_GRANT_METADATA_KEY in approval.metadata:
            raise ValueError(f"metadata[{TOOL_AUTHORIZATION_GRANT_METADATA_KEY!r}] is reserved.")
        approval = await self._private_request(approval)
        grant_id = tool_authorization_grant_id(
            session_id=approval.session_id, approval_id=approval.approval_id
        )
        actor = ToolAuthorizationGrantActor.from_resolution_actor(approval.resolved_by)
        existing = await self._store.get(grant_id)
        session_store = self._app.session_store
        session = await session_store.load(approval.session_id)
        if session is None:
            raise KeyError(f"Session not found: {approval.session_id}")
        await self._app._require_participant_execution(session, context)
        pending = await self._pending_snapshot(approval)
        if existing is not None and (
            pending is None or existing.status is not ToolAuthorizationGrantStatus.STAGED
        ):
            # A retry after the approval was consumed: the snapshot is gone, so
            # only the same resolver repeating the same options can rejoin it.
            _require_same_options(existing, grant, actor, self._rate(grant))
            request = _approval_request_with_marker(
                approval, grant_id=existing.grant_id, scope=existing.scope
            )
            if existing.status is ToolAuthorizationGrantStatus.STAGED:
                existing = await self.reconcile(grant_id)
            return PreparedToolAuthorizationGrant(grant=existing, approval_request=request)
        if pending is None:
            raise ToolAuthorizationGrantUnavailable(
                "The approval is not pending; a grant must be created before it is resolved."
            )
        now = self._clock()
        # Mandatory controls are enforced by the policy, whose offer is then not grantable.
        offer = _grantable_offer(pending, session_id=approval.session_id)
        boundary, matcher_kind, tool_name = offer.boundary, offer.matcher_kind, offer.tool_name
        if boundary is None or matcher_kind is None or tool_name is None:
            raise ToolAuthorizationGrantUnavailable("The grant offer is incomplete.")
        binding = session.invocation.resource_access
        principal = None if binding is None else (binding.authority, binding.subject)
        recorded = (
            None
            if boundary.principal is None
            else (boundary.principal.authority, boundary.principal.subject)
        )
        if recorded != principal:
            raise ToolAuthorizationGrantUnavailable(
                "The grant offer does not match the session's principal."
            )
        matcher = grant_matcher_from_arguments(
            matcher_kind,
            pending.arguments,
            pinned_argument_paths=grant.pinned_argument_paths,
        )
        if grant.scope is ToolAuthorizationGrantScopeKind.SESSION:
            scope = ToolAuthorizationGrantScope(kind=grant.scope, id=approval.session_id)
        elif boundary.workspace_id is None or not boundary.workspace_explicit:
            raise ToolAuthorizationGrantUnavailable(
                "A workspace grant requires the approved call to run in a workspace "
                "the application identified explicitly."
            )
        else:
            scope = ToolAuthorizationGrantScope(kind=grant.scope, id=boundary.workspace_id)
        created_at = now if existing is None else existing.created_at
        request = _approval_request_with_marker(approval, grant_id=grant_id, scope=scope)
        draft = ToolAuthorizationGrant(
            grant_id=grant_id,
            policy_id=offer.policy_id,
            tool_name=tool_name,
            approval_gate=offer.approval_gate,
            matcher=matcher,
            # Published arguments are only redaction-safe with a static secret scope.
            arguments_public=pending.publish_arguments
            and pending.secret_resolution_scope == "static",
            scope=scope,
            boundary=boundary,
            origin=ToolAuthorizationGrantOrigin(
                session_id=approval.session_id,
                session_instance_id=session.instance_id,
                approval_id=approval.approval_id,
                tool_round_id=approval.tool_round_id,
                tool_call_id=approval.tool_call_id,
            ),
            resolution_request_digests=(_resolution_request_digest(request),),
            created_by=actor,
            created_at=created_at,
            lifetime_seconds=grant.expires_in_seconds,
            expires_at=_expiry(created_at, grant.expires_in_seconds),
            sampled_review_rate_bps=self._rate(grant),
            justification=grant.justification,
            evidence=grant.evidence,
        )
        authorized = await self._authority.authorize_creation(
            ToolAuthorizationGrantCreation(actor=approval.resolved_by, grant=draft, offer=offer)
        )
        if authorized is not True:
            raise ToolAuthorizationGrantDenied("The actor is not authorized to create this grant.")
        stored = await self._store.stage(draft)
        return PreparedToolAuthorizationGrant(grant=stored, approval_request=request)

    async def approve_and_grant(
        self,
        approval: ToolApprovalRequest,
        grant: ToolAuthorizationGrantRequest,
        *,
        context: CollaborationAccessContext | None = None,
    ) -> AsyncIterator[Event]:
        """Remember and approve: stage the grant, resolve the approval, activate.

        Idempotent by approval identity. A failed grant write raises before the
        approval is touched. A retry after a partial attempt reuses the staged
        grant and the runtime's exactly-once approval resolution, so it cannot
        create a second grant or execute the approved call twice.
        """

        prepared = await self.prepare(approval, grant, context=context)
        grant_id = prepared.grant.grant_id
        if prepared.grant.status in {
            ToolAuthorizationGrantStatus.ABANDONED,
            ToolAuthorizationGrantStatus.REVOKED,
        }:
            raise ToolAuthorizationGrantConflict(
                f"The grant for this approval is {prepared.grant.status.value}."
            )
        from cayu.runtime._delegated_event_stream import _close_delegated_event_stream

        settled = prepared.grant.status is not ToolAuthorizationGrantStatus.STAGED
        try:
            # Own the runtime stream: every exit, including closing this stream or a
            # failed reconciliation, waits for the runtime's own cleanup to finish.
            async with _close_delegated_event_stream(
                self._app.resolve_tool_approval(prepared.approval_request, context=context)
            ) as resolution:
                async for event in resolution:
                    if not settled and event.type == EventType.TOOL_CALL_APPROVED:
                        settled = (
                            await self.reconcile(grant_id)
                        ).status is not ToolAuthorizationGrantStatus.STAGED
                    yield event
        finally:
            if not settled:
                with contextlib.suppress(Exception):
                    await self.reconcile(grant_id)

    async def reconcile(self, grant_id: str) -> ToolAuthorizationGrant:
        """Settle a staged grant from durable approval evidence.

        ACTIVE when the originating approval was granted by one of the grant's
        resolution requests; ABANDONED when it was denied, expired, or approved
        by any other request; unchanged while it is still pending.
        """

        for _attempt in range(_RECONCILE_ATTEMPTS):
            grant = await self._require(grant_id)
            if grant.status is not ToolAuthorizationGrantStatus.STAGED:
                return grant
            outcome, resolver = await self._approval_outcome(grant)
            if outcome == "approved":
                return await self._store.activate(
                    grant_id, resolved_by=resolver, activated_at=self._clock()
                )
            if outcome == "pending":
                return grant
            try:
                # Abandon only from the approval requests this classification saw.
                return await self._store.abandon(
                    grant_id,
                    reason=outcome,
                    abandoned_at=self._clock(),
                    expected_request_digests=grant.resolution_request_digests,
                )
            except ToolAuthorizationGrantConflict:
                continue
        # The grant kept changing; leave it for the next reconciliation.
        return await self._require(grant_id)

    async def reconcile_staged(
        self, *, limit: int = TOOL_AUTHORIZATION_GRANT_MAX_LIST
    ) -> tuple[ToolAuthorizationGrant, ...]:
        """Settle every staged grant, for example at startup after a crash."""

        settled: list[ToolAuthorizationGrant] = []
        query: ToolAuthorizationGrantQuery | None = ToolAuthorizationGrantQuery(
            statuses=(ToolAuthorizationGrantStatus.STAGED,), limit=limit
        )
        while query is not None:
            page = await self._store.list(query)
            settled.extend([await self.reconcile(grant.grant_id) for grant in page])
            # The cursor advances past grants that stay pending as well as settled ones.
            query = query.next_page(page)
        return tuple(settled)

    async def get(self, grant_id: str) -> ToolAuthorizationGrant | None:
        return await self._store.get(grant_id)

    async def list_grants(
        self, query: ToolAuthorizationGrantQuery | None = None
    ) -> tuple[ToolAuthorizationGrant, ...]:
        return await self._store.list(query or ToolAuthorizationGrantQuery())

    async def list_applicable(
        self,
        *,
        session_id: str,
        workspace_id: str | None = None,
        policy_id: str | None = None,
        tool_name: str | None = None,
    ) -> tuple[ToolAuthorizationGrant, ...]:
        """Active, unexpired grants scoped to this session or workspace.

        Boundary and matcher checks still run per call in the policy.
        """

        session_id = await self._app._resolve_public_session_id(session_id)
        now = self._clock()
        scopes = [(ToolAuthorizationGrantScopeKind.SESSION, session_id)]
        if workspace_id is not None:
            scopes.append((ToolAuthorizationGrantScopeKind.WORKSPACE, workspace_id))
        found: list[ToolAuthorizationGrant] = []
        for kind, scope_id in scopes:
            query: ToolAuthorizationGrantQuery | None = ToolAuthorizationGrantQuery(
                policy_id=policy_id,
                tool_name=tool_name,
                scope_kind=kind,
                scope_id=scope_id,
                statuses=(ToolAuthorizationGrantStatus.ACTIVE,),
                limit=TOOL_AUTHORIZATION_GRANT_MAX_LIST,
            )
            while query is not None:
                page = await self._store.list(query)
                found.extend(grant for grant in page if grant.is_active(now))
                query = query.next_page(page)
        return tuple(found)

    async def revoke(
        self,
        grant_id: str,
        *,
        actor: ResolutionActor | None,
        reason: str | None = None,
    ) -> ToolAuthorizationGrant:
        """Revoke a grant after ``authority`` approves the actor."""

        grant = await self._require(grant_id)
        if await self._authority.authorize_revocation(actor, grant) is not True:
            raise ToolAuthorizationGrantDenied("The actor is not authorized to revoke this grant.")
        return await self._store.revoke(
            grant_id,
            revoked_by=ToolAuthorizationGrantActor.from_resolution_actor(actor),
            reason=reason,
            revoked_at=self._clock(),
        )

    async def list_uses(
        self,
        grant_id: str,
        *,
        limit: int = 100,
        after: ToolAuthorizationGrantUseCursor | None = None,
    ) -> tuple[ToolAuthorizationGrantUse, ...]:
        return await self._store.list_uses(grant_id, limit=limit, after=after)

    async def _require(self, grant_id: str) -> ToolAuthorizationGrant:
        grant = await self._store.get(grant_id)
        if grant is None:
            raise ToolAuthorizationGrantNotFound(f"Tool authorization grant not found: {grant_id}")
        return grant

    async def _private_request(self, approval: ToolApprovalRequest) -> ToolApprovalRequest:
        app = self._app
        session_id = await app._resolve_public_session_id(approval.session_id)
        updates = {"session_id": session_id}
        for field_name in ("approval_id", "tool_round_id", "tool_call_id"):
            updates[field_name] = await app._resolve_public_action_linkage(
                session_id=session_id,
                value=getattr(approval, field_name),
                field_name=field_name,
            )
        return copy_tool_approval_request(approval).model_copy(update=updates)

    async def _pending_snapshot(self, approval: ToolApprovalRequest) -> PendingToolApproval | None:
        store = self._app.session_store
        receipt = await store.load_runtime_publication_receipt(
            approval.session_id, f"approval-close:{approval.approval_id}"
        )
        if receipt is not None:
            return None
        checkpoint = await store.load_checkpoint(approval.session_id)
        pending = self._app._pending_tool_approval_from_checkpoint(checkpoint)
        if (
            pending is None
            or pending.approval_id != approval.approval_id
            or pending.tool_round_id != approval.tool_round_id
            or pending.tool_call_id != approval.tool_call_id
        ):
            return None
        if pending.expires_at is not None and self._clock() >= pending.expires_at:
            raise ToolAuthorizationGrantUnavailable("The pending approval has expired.")
        return pending

    async def _approval_outcome(
        self, grant: ToolAuthorizationGrant
    ) -> tuple[str, ToolAuthorizationGrantActor | None]:
        """Classify the originating approval from one coherent view of durable evidence.

        The closure receipt is read first, so the events read next include every
        approval it closed, and again last. If the receipt changed meanwhile, the
        approval was being published concurrently and the grant stays staged.
        """

        from cayu.runtime._session_queries import query_all_event_records
        from cayu.sessions import _pending_approval_reader as pending_approval_reader

        origin = grant.origin
        store = self._app.session_store
        receipt_key = f"approval-close:{origin.approval_id}"
        if await store.load(origin.session_id) is None:
            return "session_unavailable", None
        receipt = await store.load_runtime_publication_receipt(origin.session_id, receipt_key)
        records = await query_all_event_records(
            store,
            EventQuery(
                session_id=origin.session_id,
                event_types=(
                    EventType.TOOL_CALL_APPROVED,
                    EventType.TOOL_CALL_APPROVAL_DENIED,
                    EventType.TOOL_CALL_APPROVAL_EXPIRED,
                ),
                limit=500,
            ),
        )
        checkpoint = await store.load_checkpoint(origin.session_id)
        latest = await store.load_runtime_publication_receipt(origin.session_id, receipt_key)
        if (receipt is None) != (latest is None) or (
            receipt is not None and latest is not None and receipt.intent != latest.intent
        ):
            return "pending", None
        closures = [
            record.event
            for record in records
            if record.event.payload.get("approval_id") == origin.approval_id
            and record.event.payload.get("tool_call_id") == origin.tool_call_id
        ]
        approved = next((e for e in closures if e.type == EventType.TOOL_CALL_APPROVED), None)
        digest: str | None = None
        if receipt is not None and receipt.intent.get("approval_id") == origin.approval_id:
            value = receipt.intent.get("resolution_request_digest")
            digest = value if type(value) is str else None
        else:
            intent = pending_approval_reader.approval_resolution_intent_from_checkpoint(checkpoint)
            if intent is not None and intent.approval_id == origin.approval_id:
                digest = intent.resolution_request_digest
        if approved is not None:
            if digest is None:
                return "pending", None
            if digest not in grant.resolution_request_digests:
                return "approved_without_grant", None
            return "approved", _event_actor(approved)
        if any(e.type == EventType.TOOL_CALL_APPROVAL_EXPIRED for e in closures):
            return "approval_expired", None
        if closures or receipt is not None:
            return "approval_denied", None
        pending = self._app._pending_tool_approval_from_checkpoint(checkpoint)
        if pending is None or pending.approval_id != origin.approval_id:
            # The approval left the checkpoint without a durable grant decision.
            return "approval_unavailable", None
        if pending.expires_at is not None and self._clock() >= pending.expires_at:
            return "approval_expired", None
        return "pending", None


def _grantable_offer(
    pending: PendingToolApproval, *, session_id: str
) -> ToolAuthorizationGrantOffer:
    raw = pending.metadata.get(TOOL_AUTHORIZATION_GRANT_OFFER_METADATA_KEY)
    if raw is None:
        raise ToolAuthorizationGrantUnavailable(
            "The pending approval was not produced by a grant policy."
        )
    try:
        offer = ToolAuthorizationGrantOffer.model_validate(raw)
    except Exception:
        raise ToolAuthorizationGrantUnavailable(
            "The pending approval carries an invalid grant offer."
        ) from None
    if not offer.grantable:
        raise ToolAuthorizationGrantUnavailable(
            f"The pending approval cannot be granted: {offer.reason}."
        )
    gating = [call for call in pending.tool_calls if call.tool_call_id == pending.tool_call_id]
    boundary = offer.boundary
    if (
        boundary is None
        or offer.tool_name != pending.tool_name
        or offer.session_id != session_id
        or boundary.agent_name != pending.agent_name
        or boundary.environment_name != pending.environment_name
        or boundary.workspace_id != pending.workspace_id
        or len(gating) != 1
        or gating[0].policy_decision not in {None, "require_approval"}
    ):
        raise ToolAuthorizationGrantUnavailable(
            "The grant offer does not match its pending approval."
        )
    return offer


def _require_same_options(
    existing: ToolAuthorizationGrant,
    grant: ToolAuthorizationGrantRequest,
    actor: ToolAuthorizationGrantActor | None,
    sampled_review_rate_bps: int,
) -> None:
    pinned: Iterable[str] = (
        existing.matcher.pinned_arguments
        if isinstance(existing.matcher, ToolCallGrantMatcher)
        else ()
    )
    if (
        existing.scope.kind is not grant.scope
        or existing.lifetime_seconds != grant.expires_in_seconds
        or existing.sampled_review_rate_bps != sampled_review_rate_bps
        or tuple(pinned) != grant.pinned_argument_paths
        or existing.justification != grant.justification
        or existing.evidence != grant.evidence
        or existing.created_by != actor
    ):
        raise ToolAuthorizationGrantConflict(
            "A different grant was already requested for this approval."
        )


def _approval_request_with_marker(
    approval: ToolApprovalRequest, *, grant_id: str, scope: ToolAuthorizationGrantScope
) -> ToolApprovalRequest:
    """Bind the approval request to the grant it activates.

    The marker is part of the request digest the runtime records when it
    claims the approval, and the approval event's audit metadata.
    """

    metadata = copy_durable_metadata(
        {
            **approval.metadata,
            TOOL_AUTHORIZATION_GRANT_METADATA_KEY: {
                "grant_id": grant_id,
                "scope_kind": scope.kind.value,
                "scope_id": scope.id,
            },
        }
    )
    return copy_tool_approval_request(approval).model_copy(update={"metadata": metadata})


def _resolution_request_digest(request: ToolApprovalRequest) -> str:
    from cayu.runtime import _approval_support as approval_support

    return approval_support.approval_resolution_request_digest(request)


def _expiry(created_at: datetime, lifetime_seconds: int | None) -> datetime | None:
    if lifetime_seconds is None:
        return None
    return normalize_grant_timestamp(created_at) + timedelta(seconds=lifetime_seconds)


def _event_actor(event: Event) -> ToolAuthorizationGrantActor | None:
    payload = event.payload.get("resolved_by")
    if not isinstance(payload, dict):
        return None
    try:
        return ToolAuthorizationGrantActor.model_validate(
            {key: payload.get(key) for key in ("subject", "tenant", "source")}
        )
    except Exception:
        return None


__all__ = [
    "PreparedToolAuthorizationGrant",
    "ToolAuthorizationGrantRequest",
    "ToolAuthorizationGrantService",
]
