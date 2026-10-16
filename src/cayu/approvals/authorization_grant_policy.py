"""A ToolPolicy composition that consults durable authorization grants."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from datetime import datetime
from types import MappingProxyType
from typing import Any

from cayu._clock import utc_clock
from cayu._validation import copy_durable_metadata, require_clean_nonblank
from cayu.approvals.authorization_grants import (
    TOOL_APPROVAL_GATE_METADATA_KEY,
    TOOL_AUTHORIZATION_GRANT_MAX_BOUNDARY_LABELS,
    TOOL_AUTHORIZATION_GRANT_MAX_LIST,
    TOOL_AUTHORIZATION_GRANT_METADATA_KEY,
    TOOL_AUTHORIZATION_GRANT_OFFER_METADATA_KEY,
    TOOL_MANDATORY_APPROVAL_METADATA_KEY,
    ToolAuthorizationGrant,
    ToolAuthorizationGrantBoundary,
    ToolAuthorizationGrantControl,
    ToolAuthorizationGrantMatcherKind,
    ToolAuthorizationGrantOffer,
    ToolAuthorizationGrantPrincipal,
    ToolAuthorizationGrantQuery,
    ToolAuthorizationGrantScopeKind,
    ToolAuthorizationGrantStatus,
    ToolAuthorizationGrantStore,
    ToolAuthorizationGrantUse,
    ToolAuthorizationGrantUseDecision,
    copy_mandatory_controls,
    is_mandatory_control,
    tool_arguments_digest,
    tool_authorization_grant_sample_bucket,
)
from cayu.runtime.execution_identity import (
    ExecutionProfileBehaviorIdentity,
    copy_execution_profile_behavior_identity,
)
from cayu.tools.policy import (
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
)

_BASE_REASON = "Tool requires approval unless an authorization grant applies."
_RESERVED_METADATA_KEYS = frozenset(
    {TOOL_AUTHORIZATION_GRANT_METADATA_KEY, TOOL_AUTHORIZATION_GRANT_OFFER_METADATA_KEY}
)


class ToolAuthorizationGrantPolicy(ToolPolicy):
    """Turn a REQUIRE_APPROVAL decision into ALLOW when an active grant applies.

    ``then`` decides first (every call requires approval when omitted). ALLOW
    and DENY pass through unchanged, so a grant can only stand in for a person's
    approval and never overrides a denial. For a REQUIRE_APPROVAL decision on a
    tool listed in ``grantable``:

    - a mandatory control (a ``(tool_name, approval_gate)`` entry in
      ``mandatory_controls``, or a result marked with ``cayu:mandatory_approval``)
      always requires approval
      and is recorded as not grantable;
    - otherwise the first active, unexpired grant whose policy, tool, gate,
      scope, boundary, and matcher all apply is selected, session grants before
      workspace grants, each ordered by ``(created_at, grant_id)``;
    - a selected grant's deterministic sample bucket decides whether the call
      still goes to sampled review. Durable use evidence is written before
      either decision is returned; if it cannot be written, approval is required.

    When no grant applies, the REQUIRE_APPROVAL result carries a grant offer in
    its metadata. The runtime persists it on the pending approval, which is how
    grant creation learns the call's class of decision and boundary without
    trusting a client. The boundary binds the session's resource-access
    principal, so grants never cross the identities trusted access control
    assigns. ``boundary_labels`` names session label keys that must also match
    exactly; a session missing one of them is never grantable. Clients can set
    labels, so they narrow a grant but are not an authority boundary. Workspace
    grants apply only to a workspace the application named explicitly.
    Reserved grant metadata in ``then`` results is dropped.
    """

    def __init__(
        self,
        *,
        policy_id: str,
        store: ToolAuthorizationGrantStore,
        grantable: Mapping[str, ToolAuthorizationGrantMatcherKind | str],
        then: ToolPolicy | None = None,
        mandatory_controls: Iterable[ToolAuthorizationGrantControl] = (),
        boundary_labels: Iterable[str] = (),
        clock: Callable[[], datetime] | None = None,
        execution_profile_identity: ExecutionProfileBehaviorIdentity | None = None,
    ) -> None:
        if not isinstance(store, ToolAuthorizationGrantStore):
            raise TypeError("store must be a ToolAuthorizationGrantStore.")
        if then is not None and not isinstance(then, ToolPolicy):
            raise TypeError("then must be a ToolPolicy.")
        if not isinstance(grantable, Mapping) or not grantable:
            raise ValueError("grantable must map at least one tool name to a matcher kind.")
        if isinstance(boundary_labels, str):
            raise TypeError("boundary_labels must be an iterable of label keys.")
        self._policy_id = require_clean_nonblank(policy_id, "policy_id")
        self._store = store
        self._then = then
        self._grantable: dict[str, ToolAuthorizationGrantMatcherKind] = {
            require_clean_nonblank(name, "tool_name"): ToolAuthorizationGrantMatcherKind(kind)
            for name, kind in grantable.items()
        }
        self._mandatory_controls = copy_mandatory_controls(mandatory_controls)
        labels = tuple(dict.fromkeys(require_clean_nonblank(k, "label") for k in boundary_labels))
        if len(labels) > TOOL_AUTHORIZATION_GRANT_MAX_BOUNDARY_LABELS:
            raise ValueError("Too many boundary labels.")
        self._boundary_labels = labels
        self._clock = utc_clock(clock)
        self._execution_profile_identity = copy_execution_profile_behavior_identity(
            execution_profile_identity
        )

    @property
    def policy_id(self) -> str:
        return self._policy_id

    @property
    def store(self) -> ToolAuthorizationGrantStore:
        return self._store

    @property
    def then(self) -> ToolPolicy | None:
        return self._then

    @property
    def grantable(self) -> Mapping[str, ToolAuthorizationGrantMatcherKind]:
        return MappingProxyType(self._grantable)

    @property
    def mandatory_controls(self) -> tuple[ToolAuthorizationGrantControl, ...]:
        return self._mandatory_controls

    @property
    def boundary_labels(self) -> tuple[str, ...]:
        return self._boundary_labels

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity | None:
        return copy_execution_profile_behavior_identity(self._execution_profile_identity)

    def _execution_profile_material(self) -> dict[str, object]:
        """Behavior inputs only; grant state is durable data, not policy identity."""

        return {
            "policy_id": self._policy_id,
            "grantable": {name: kind.value for name, kind in sorted(self._grantable.items())},
            "mandatory_controls": [
                control.model_dump(mode="json")
                for control in sorted(
                    self._mandatory_controls,
                    key=lambda item: (item.tool_name, item.approval_gate or ""),
                )
            ],
            "boundary_labels": list(self._boundary_labels),
        }

    def boundary_for(self, request: ToolPolicyRequest) -> ToolAuthorizationGrantBoundary | None:
        """The request's authorization boundary, or ``None`` when a label is missing."""

        labels: dict[str, str] = {}
        for key in self._boundary_labels:
            value = request.session.labels.get(key)
            if value is None:
                return None
            labels[key] = value
        binding = request.session.invocation.resource_access
        return ToolAuthorizationGrantBoundary(
            agent_name=request.agent.name,
            environment_name=request.environment_name,
            workspace_id=request.workspace_id,
            workspace_explicit=request.workspace_id is not None and request.workspace_id_explicit,
            principal=(
                None
                if binding is None
                else ToolAuthorizationGrantPrincipal(
                    authority=binding.authority, subject=binding.subject
                )
            ),
            labels=labels,
        )

    async def applicable_grants(
        self, request: ToolPolicyRequest, *, approval_gate: str | None = None
    ) -> tuple[ToolAuthorizationGrant, ...]:
        """Active grants that would authorize ``request``, in selection order."""

        boundary = self.boundary_for(request)
        if boundary is None or request.tool_name not in self._grantable:
            return ()
        now = self._clock()
        scopes = [(ToolAuthorizationGrantScopeKind.SESSION, request.session.id)]
        if boundary.workspace_explicit and boundary.workspace_id is not None:
            scopes.append((ToolAuthorizationGrantScopeKind.WORKSPACE, boundary.workspace_id))
        applicable: list[ToolAuthorizationGrant] = []
        for scope_kind, scope_id in scopes:
            query: ToolAuthorizationGrantQuery | None = ToolAuthorizationGrantQuery(
                policy_id=self._policy_id,
                tool_name=request.tool_name,
                scope_kind=scope_kind,
                scope_id=scope_id,
                statuses=(ToolAuthorizationGrantStatus.ACTIVE,),
                limit=TOOL_AUTHORIZATION_GRANT_MAX_LIST,
            )
            # Expired grants keep their stored ACTIVE status, so read every page.
            while query is not None:
                page = await self._store.list(query)
                applicable.extend(
                    grant
                    for grant in page
                    if self._grant_applies(grant, request, boundary, approval_gate, now)
                )
                query = query.next_page(page)
        return tuple(applicable)

    def _grant_applies(
        self,
        grant: ToolAuthorizationGrant,
        request: ToolPolicyRequest,
        boundary: ToolAuthorizationGrantBoundary,
        approval_gate: str | None,
        now: datetime,
    ) -> bool:
        expected_kind = self._grantable.get(request.tool_name)
        return (
            grant.is_active(now)
            and (
                grant.scope.kind is not ToolAuthorizationGrantScopeKind.SESSION
                or grant.origin.session_instance_id == request.session.instance_id
            )
            and grant.policy_id == self._policy_id
            and grant.tool_name == request.tool_name
            and grant.approval_gate == approval_gate
            and grant.boundary == boundary
            and grant.matcher.kind == expected_kind
            and grant.matcher.matches(request.arguments)
        )

    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        base = await self._base_decision(request)
        if base.decision is ToolPolicyDecision.DENY:
            return base
        gate = base.metadata.get(TOOL_APPROVAL_GATE_METADATA_KEY)
        mandatory = is_mandatory_control(
            self._mandatory_controls, tool_name=request.tool_name, approval_gate=gate
        ) or (
            base.decision is ToolPolicyDecision.REQUIRE_APPROVAL
            and base.metadata.get(TOOL_MANDATORY_APPROVAL_METADATA_KEY) is True
        )
        if mandatory:
            # A mandatory control always asks a person, whatever the downstream decided.
            required = (
                base
                if base.decision is ToolPolicyDecision.REQUIRE_APPROVAL
                else base.model_copy(
                    update={
                        "decision": ToolPolicyDecision.REQUIRE_APPROVAL,
                        "reason": "A mandatory control requires approval.",
                    }
                )
            )
            return self._with_offer(required, grantable=False, reason="mandatory_control")
        if base.decision is not ToolPolicyDecision.REQUIRE_APPROVAL:
            return base
        matcher_kind = self._grantable.get(request.tool_name)
        if matcher_kind is None:
            return base
        if gate is not None and (type(gate) is not str or not gate.strip()):
            return self._with_offer(base, grantable=False, reason="invalid_approval_gate")
        boundary = self.boundary_for(request)
        if boundary is None:
            return self._with_offer(base, grantable=False, reason="boundary_unavailable")
        try:
            grants = await self.applicable_grants(request, approval_gate=gate)
        except Exception:
            # An unreadable store can never authorize; a person decides instead.
            return self._with_offer(base, grantable=False, reason="grant_store_unavailable")
        if not grants:
            return self._with_offer(
                base,
                grantable=True,
                tool_name=request.tool_name,
                approval_gate=gate,
                matcher_kind=matcher_kind,
                session_id=request.session.id,
                boundary=boundary,
            )
        grant = grants[0]
        bucket = tool_authorization_grant_sample_bucket(
            grant_id=grant.grant_id,
            session_id=request.session.id,
            tool_call_id=request.tool_call_id,
            tool_round_id=request.tool_round_id,
        )
        sampled = bucket < grant.sampled_review_rate_bps
        try:
            use = await self._store.record_use(
                ToolAuthorizationGrantUse(
                    grant_id=grant.grant_id,
                    policy_id=self._policy_id,
                    scope=grant.scope,
                    session_id=request.session.id,
                    session_instance_id=request.session.instance_id,
                    tool_round_id=request.tool_round_id,
                    tool_call_id=request.tool_call_id,
                    tool_name=request.tool_name,
                    arguments_digest=tool_arguments_digest(request.arguments),
                    decision=(
                        ToolAuthorizationGrantUseDecision.SAMPLED_REVIEW
                        if sampled
                        else ToolAuthorizationGrantUseDecision.ALLOWED
                    ),
                    sample_bucket=bucket,
                    sampled_review_rate_bps=grant.sampled_review_rate_bps,
                    recorded_at=self._clock(),
                )
            )
        except Exception:
            return self._with_offer(base, grantable=False, reason="grant_evidence_unavailable")
        evidence = {
            **grant.evidence_payload(),
            "decision": use.decision.value,
            "sample_bucket": use.sample_bucket,
            "sampled_review_rate_bps": use.sampled_review_rate_bps,
        }
        if use.decision is ToolAuthorizationGrantUseDecision.SAMPLED_REVIEW:
            sampled_result = base.model_copy(
                update={
                    "reason": "Authorization grant selected this call for sampled review.",
                    "metadata": {
                        **base.metadata,
                        TOOL_AUTHORIZATION_GRANT_METADATA_KEY: evidence,
                    },
                }
            )
            return self._with_offer(sampled_result, grantable=False, reason="sampled_review")
        return ToolPolicyResult(
            decision=ToolPolicyDecision.ALLOW,
            reason="Authorized by a durable tool authorization grant.",
            metadata={TOOL_AUTHORIZATION_GRANT_METADATA_KEY: evidence},
        )

    async def _base_decision(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        if self._then is None:
            return ToolPolicyResult(
                decision=ToolPolicyDecision.REQUIRE_APPROVAL, reason=_BASE_REASON
            )
        result = await self._then.authorize(request.model_copy(deep=True))
        if type(result) is not ToolPolicyResult:
            raise TypeError("then.authorize() must return a ToolPolicyResult.")
        result = ToolPolicyResult.model_validate(result.model_dump())
        # Grant evidence and offers are written only by this policy.
        if _RESERVED_METADATA_KEYS.isdisjoint(result.metadata):
            return result
        metadata = {
            key: value
            for key, value in result.metadata.items()
            if key not in _RESERVED_METADATA_KEYS
        }
        return result.model_copy(update={"metadata": metadata})

    def _with_offer(self, result: ToolPolicyResult, **offer: Any) -> ToolPolicyResult:
        payload = ToolAuthorizationGrantOffer(policy_id=self._policy_id, **offer).model_dump(
            mode="json", exclude_none=True
        )
        metadata = copy_durable_metadata(
            {**result.metadata, TOOL_AUTHORIZATION_GRANT_OFFER_METADATA_KEY: payload}
        )
        return result.model_copy(update={"metadata": metadata})


__all__ = ["ToolAuthorizationGrantPolicy"]
