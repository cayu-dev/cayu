"""Business approval adapter — tier routing and three-outcome approvals.

Cayu's approval primitive is deliberately small: one pending tool approval,
resolved with a binary approve/deny. Real business approvals usually need
more — a routing tier computed per request (only the matching tier may
resolve) and a third outcome ("approve with conditions"). This module keeps
the primitive untouched and layers those semantics on top of its public
surface:

- a versioned routing carrier a ``ToolPolicy`` attaches when it pauses a call
  (``business_approval_routing_metadata`` / ``TieredApprovalPolicy``);
- a gated resolver mapping approve/condition/decline onto the binary decision
  while stamping the business outcome into the resolution metadata
  (``resolve_business_approval``);
- an audit projection pairing approval events into business records
  (``business_approval_audit``).

The tier gate runs in the calling process. It is an application affordance,
not a security boundary: callers with direct access to
``CayuApp.resolve_tool_approval`` or ``POST /api/tool-approvals/resolve``
bypass it, and ``approver_id``/``approver_tier`` are caller-asserted. The
audit projection records what callers asserted — it does not authenticate
it: unstamped raw-primitive resolutions show ``outcome=None``, while a
caller at the same trust level can also write the stamp by hand through the
raw primitive. Authenticate approvers and guard the raw route in the product
layer that owns them.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from cayu._validation import (
    copy_durable_json_value,
    require_durable_clean_nonblank,
    require_durable_nonblank,
)
from cayu.core.events import Event, EventType
from cayu.runtime import _approval_support as approval_support
from cayu.runtime.approvals import (
    PendingToolApproval,
    ResolutionActor,
    ToolApprovalDecision,
    ToolApprovalRequest,
    ToolPolicyEvidence,
    pending_tool_call_for_approval_event,
)
from cayu.runtime.budgets import BudgetLimit
from cayu.runtime.execution_units import ToolRoundIdentity
from cayu.runtime.loop_policies import LoopPolicy
from cayu.runtime.retry_policy import RetryPolicy
from cayu.runtime.stop_policy import RunLimits
from cayu.runtime.structured_output import StructuredOutputSpec
from cayu.runtime.tool_policy import (
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
)

if TYPE_CHECKING:
    from cayu.core.thinking import ThinkingConfig
    from cayu.runtime.app import CayuApp

BUSINESS_APPROVAL_ROUTING_METADATA_KEY = "cayu:business_approval_routing"
BUSINESS_APPROVAL_ROUTING_KIND = "cayu.business_approval.routing.v1"
BUSINESS_APPROVAL_RESOLUTION_METADATA_KEY = (
    approval_support.BUSINESS_APPROVAL_RESOLUTION_METADATA_KEY
)
BUSINESS_APPROVAL_RESOLUTION_KIND = "cayu.business_approval.v1"


class BusinessApprovalOutcome(StrEnum):
    APPROVED = "approved"
    CONDITIONED = "conditioned"
    DECLINED = "declined"


class BusinessApprovalResolutionState(StrEnum):
    """Durable lifecycle state of one business approval."""

    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    BLOCKED = "blocked"


class BusinessApprovalError(Exception):
    """Base class for business-approval adapter failures."""


class BusinessApprovalRoutingMissing(BusinessApprovalError):
    """The pending approval carries no (valid) business-approval routing.

    Raised when a pause was created without ``business_approval_routing_metadata``
    (for example by a plain policy) or the carrier payload is malformed. The
    adapter refuses rather than resolving ungated — resolve such approvals with
    ``CayuApp.resolve_tool_approval`` directly.
    """


class BusinessApprovalTierMismatch(BusinessApprovalError):
    """The asserted approver tier may not resolve this approval."""

    def __init__(self, *, approver_tier: str, required_tier: str, chain: tuple[str, ...]) -> None:
        self.approver_tier = approver_tier
        self.required_tier = required_tier
        self.chain = chain
        super().__init__(
            f"Approver tier {approver_tier!r} may not resolve an approval routed to "
            f"{required_tier!r} (chain: {' -> '.join(chain)})."
        )


class BusinessApprovalRouting(BaseModel):
    """Product-computed routing for one business approval.

    ``chain`` is ordered lowest to highest authority; ``required_tier`` is the
    minimum tier that may resolve. ``metadata`` carries product extras (package
    id, amount, cost center, ...) for the approval UI and the audit trail.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    kind: str = BUSINESS_APPROVAL_ROUTING_KIND
    required_tier: str
    chain: tuple[str, ...]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("kind")
    @classmethod
    def validate_kind(cls, value: str) -> str:
        if value != BUSINESS_APPROVAL_ROUTING_KIND:
            raise ValueError(
                f"Business approval routing kind must be {BUSINESS_APPROVAL_ROUTING_KIND!r}."
            )
        return value

    @field_validator("required_tier")
    @classmethod
    def validate_required_tier(cls, value: str) -> str:
        return require_durable_clean_nonblank(value, "required_tier")

    @field_validator("chain", mode="before")
    @classmethod
    def validate_chain(cls, value: Any) -> tuple[str, ...]:
        if isinstance(value, str) or not isinstance(value, Sequence):
            raise ValueError("`chain` must be a sequence of tier names.")
        tiers = tuple(require_durable_clean_nonblank(tier, "chain") for tier in value)
        if not tiers:
            raise ValueError("`chain` cannot be empty.")
        if len(set(tiers)) != len(tiers):
            raise ValueError("`chain` cannot contain duplicate tiers.")
        return tiers

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: Any) -> dict[str, Any]:
        return copy_durable_json_value(value, "metadata")

    @model_validator(mode="after")
    def validate_required_tier_in_chain(self) -> BusinessApprovalRouting:
        if self.required_tier not in self.chain:
            raise ValueError(f"required_tier {self.required_tier!r} must be a member of the chain.")
        return self


def business_approval_routing_metadata(
    *,
    required_tier: str,
    chain: Sequence[str],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the ``ToolPolicyResult.metadata`` carrier for a routed approval.

    Merge the returned dict into the metadata of a
    ``ToolPolicyResult(decision=REQUIRE_APPROVAL, ...)``; the routing then
    persists on the pending approval and surfaces to approval UIs in the
    ``tool.call.approval_requested`` event under
    ``payload["approval"]["metadata"]``.
    """
    routing = BusinessApprovalRouting(
        required_tier=required_tier,
        chain=tuple(chain),
        metadata=dict(metadata) if metadata is not None else {},
    )
    return {BUSINESS_APPROVAL_ROUTING_METADATA_KEY: routing.model_dump(mode="json")}


class TieredApprovalPolicy(ToolPolicy):
    """Pause tool calls for business approval with a product-computed routing.

    Tier computation stays product-owned: ``compute_routing`` receives the
    ``ToolPolicyRequest`` and returns a ``BusinessApprovalRouting`` (pause for
    approval at that tier) or ``None`` (allow without approval). The callback
    may be sync or async — a product implementation typically consults its own
    rules or database. It can run more than once for the same call: the
    runtime re-authorizes when a ``before_tool_call`` hook modifies arguments
    (marked with ``request.metadata[TOOL_POLICY_REAUTHORIZATION_METADATA_KEY]``),
    so keep the callback deterministic and side-effect free; a
    REQUIRE_APPROVAL verdict on re-authorized arguments blocks the call
    instead of pausing again. ``expires_in_seconds`` bounds how long the
    resulting approval demand stays valid, exactly like
    ``AlwaysRequireApprovalToolPolicy(expires_in_seconds=...)``.
    """

    def __init__(
        self,
        *,
        compute_routing: Callable[
            [ToolPolicyRequest],
            BusinessApprovalRouting | None | Awaitable[BusinessApprovalRouting | None],
        ],
        reason: str | None = None,
        expires_in_seconds: float | None = None,
    ) -> None:
        if not callable(compute_routing):
            raise TypeError("compute_routing must be callable.")
        self._compute_routing = compute_routing
        self._reason = require_durable_nonblank(reason, "reason") if reason is not None else None
        self._expires_in_seconds = expires_in_seconds

    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        routing = self._compute_routing(request)
        if inspect.isawaitable(routing):
            routing = await routing
        if routing is None:
            return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)
        if type(routing) is not BusinessApprovalRouting:
            raise TypeError("compute_routing must return a BusinessApprovalRouting or None.")
        return ToolPolicyResult(
            decision=ToolPolicyDecision.REQUIRE_APPROVAL,
            reason=self._reason,
            metadata={
                BUSINESS_APPROVAL_ROUTING_METADATA_KEY: routing.model_dump(mode="json"),
            },
            approval_expires_in_seconds=self._expires_in_seconds,
        )


def business_approval_routing(pending: PendingToolApproval) -> BusinessApprovalRouting:
    """Read the routing carrier off a pending approval.

    Raises ``BusinessApprovalRoutingMissing`` when the pause was created
    without the carrier or the carrier is malformed — the tier gate never
    silently degrades into an ungated resolve.
    """
    if type(pending) is not PendingToolApproval:
        raise TypeError("Business approval routing requires a PendingToolApproval.")
    value = pending.metadata.get(BUSINESS_APPROVAL_ROUTING_METADATA_KEY)
    if value is None:
        raise BusinessApprovalRoutingMissing(
            f"Pending approval {pending.approval_id} carries no business approval routing; "
            "resolve it with CayuApp.resolve_tool_approval directly."
        )
    try:
        return BusinessApprovalRouting.model_validate(value)
    except ValidationError as exc:
        raise BusinessApprovalRoutingMissing(
            f"Pending approval {pending.approval_id} carries a malformed business approval "
            f"routing: {exc}"
        ) from exc


async def resolve_business_approval(
    app: CayuApp,
    *,
    session_id: str,
    approval_id: str,
    approver_id: str,
    approver_tier: str,
    outcome: BusinessApprovalOutcome | str,
    condition_text: str | None = None,
    reason: str | None = None,
    metadata: dict[str, Any] | None = None,
    resolved_by: ResolutionActor | None = None,
    max_steps: int | None = None,
    limits: RunLimits | None = None,
    budget_limits: tuple[BudgetLimit, ...] | None = None,
    retry_policy: RetryPolicy | None = None,
    structured_output: StructuredOutputSpec | None = None,
    thinking: ThinkingConfig | None = None,
    loop_policies: tuple[LoopPolicy, ...] = (),
) -> AsyncIterator[Event]:
    """Resolve a routed business approval and return the resumed event stream.

    Enforces the tier gate against the routing persisted on the session's
    pending-approval checkpoint — never a caller-supplied value —
    (``approver_tier`` must sit at or above ``required_tier`` in the
    ordered chain), maps the business outcome onto the binary primitive
    (``APPROVED``/``CONDITIONED`` resume as approve — ``CONDITIONED`` requires
    ``condition_text``; ``DECLINED`` resumes as deny), stamps the business
    outcome into the resolution metadata for the audit trail, and delegates to
    ``CayuApp.resolve_tool_approval``.

    This is a coroutine returning the event iterator: gate failures raise at
    ``await`` time, before any event is consumed —
    ``events = await resolve_business_approval(...)`` then iterate. The
    approved tool sees the stamped record via
    ``ToolContext.metadata["cayu:business_approval"]``; the model does not see
    approve-path metadata, so a condition that must steer the model has to be
    surfaced by the tool's own result. The gate is an application affordance,
    not a security boundary: ``approver_id``/``approver_tier`` are
    caller-asserted, and direct primitive callers bypass the adapter entirely
    (unstamped resolutions show ``outcome=None`` in the audit; a same-trust
    caller can also forge the stamp — authenticate approvers product-side and
    pass ``resolved_by`` (a ``ResolutionActor`` from your auth layer) so the
    typed provenance the runtime stamps sits next to the business assertion).
    The primitive independently reloads and re-validates the pending approval
    when the returned stream is first iterated, so a resolution race lost
    after this gate surfaces as the primitive's own status-conflict error.
    Approval expiry composes the same way: a tier-authorized resolution whose
    approval window already closed is coerced by the runtime to a denial —
    no typed error; the stream carries ``tool.call.approval_expired`` and
    denial events with ``expired: true``, and the audit record shows
    ``expired=True`` next to the asserted outcome.
    """
    session_id = require_durable_clean_nonblank(session_id, "session_id")
    approval_id = require_durable_clean_nonblank(approval_id, "approval_id")
    approver_id = require_durable_clean_nonblank(approver_id, "approver_id")
    approver_tier = require_durable_clean_nonblank(approver_tier, "approver_tier")
    outcome = BusinessApprovalOutcome(outcome)
    if outcome is BusinessApprovalOutcome.CONDITIONED:
        condition_text = require_durable_nonblank(
            condition_text if condition_text is not None else "", "condition_text"
        )
    elif condition_text is not None:
        raise ValueError("condition_text is only valid with the CONDITIONED outcome.")

    # Load the persisted pending approval and gate against ITS routing — the
    # tier gate is adapter-only (the primitive does not know tiers), so its
    # input must come from trusted checkpoint state, never from the caller.
    checkpoint = await app._runtime_session_store.load_checkpoint(session_id)
    pending = app._pending_tool_approval_from_checkpoint(
        checkpoint,
        consume_on_rejection=True,
    )
    if pending is None:
        raise RuntimeError("Session has no pending tool approval.")
    if pending.approval_id != approval_id:
        raise ValueError(f"Tool approval id does not match pending approval: {approval_id}")

    routing = business_approval_routing(pending)
    if approver_tier not in routing.chain or routing.chain.index(
        approver_tier
    ) < routing.chain.index(routing.required_tier):
        raise BusinessApprovalTierMismatch(
            approver_tier=approver_tier,
            required_tier=routing.required_tier,
            chain=routing.chain,
        )

    if outcome is BusinessApprovalOutcome.DECLINED:
        decision = ToolApprovalDecision.DENY
        if reason is None:
            reason = f"Business approval declined by {approver_id} ({approver_tier})."
    else:
        decision = ToolApprovalDecision.APPROVE

    if metadata is not None and type(metadata) is not dict:
        raise TypeError("metadata must be a dict.")
    resolution_metadata = copy_durable_json_value(metadata or {}, "metadata")
    if BUSINESS_APPROVAL_RESOLUTION_METADATA_KEY in resolution_metadata:
        raise ValueError(
            f"metadata must not contain the reserved key "
            f"{BUSINESS_APPROVAL_RESOLUTION_METADATA_KEY!r}."
        )
    resolution_metadata[BUSINESS_APPROVAL_RESOLUTION_METADATA_KEY] = {
        "kind": BUSINESS_APPROVAL_RESOLUTION_KIND,
        "outcome": outcome.value,
        "condition_text": condition_text,
        "approver_id": approver_id,
        "approver_tier": approver_tier,
        "required_tier": routing.required_tier,
        "chain": list(routing.chain),
    }

    request = ToolApprovalRequest(
        session_id=session_id,
        approval_id=approval_id,
        tool_round_id=pending.tool_round_id,
        tool_call_id=pending.tool_call_id,
        decision=decision,
        reason=reason,
        metadata=resolution_metadata,
        resolved_by=resolved_by,
        max_steps=max_steps,
        limits=limits,
        budget_limits=budget_limits,
        retry_policy=retry_policy,
        structured_output=structured_output,
        thinking=thinking,
        loop_policies=loop_policies,
    )
    return app.resolve_tool_approval(request)


@dataclass(frozen=True)
class BusinessApprovalRecord:
    """One business approval, projected from the session's approval events.

    ``resolution_state`` distinguishes a pending request, a binary runtime
    decision, and a non-authorizing recovery acknowledgement. ``decision`` is
    the raw Cayu decision (``None`` while pending or safely blocked);
    ``outcome`` is the business outcome stamped by the adapter (``None`` while
    pending, and ``None`` on unstamped resolutions made through the raw
    primitive). ``resolved_by`` is the typed ``ResolutionActor`` provenance
    the runtime stamped on the resolution event, when the resolver supplied
    one — the authenticated who/how next to the caller-asserted
    ``approver_id``/``approver_tier``. ``expired`` is True when the runtime's
    approval TTL closed on the attempt and coerced it to a denial: the
    stamped business fields then record what the approver asserted, while
    ``decision`` records the expiry-coerced outcome (so ``decision=DENY``
    with ``outcome=approved`` reads as "approved too late").
    """

    approval_id: str
    session_id: str
    tool_call_id: str
    tool_name: str
    agent_name: str
    routing: BusinessApprovalRouting | None
    resolution_state: BusinessApprovalResolutionState
    decision: ToolApprovalDecision | None
    outcome: BusinessApprovalOutcome | None
    condition_text: str | None
    approver_id: str | None
    approver_tier: str | None
    resolved_by: ResolutionActor | None
    reason: str | None
    expired: bool
    requested_at: datetime | None
    resolved_at: datetime | None

    @property
    def pending(self) -> bool:
        return self.resolution_state is BusinessApprovalResolutionState.PENDING

    @property
    def resolved_via_adapter(self) -> bool:
        return self.outcome is not None


def _approval_request_candidate_ids(event: Event) -> set[str]:
    """Retain raw ids so a malformed duplicate still poisons its approval record."""
    nested = event.payload.get("approval")
    nested_id = nested.get("approval_id") if type(nested) is dict else None
    return {value for value in (event.payload.get("approval_id"), nested_id) if type(value) is str}


def business_approval_audit(events: Iterable[Event]) -> list[BusinessApprovalRecord]:
    """Project approval events into business approval records.

    Pass the output of ``session_store.load_events(...)`` or an
    ``EventQuery`` filtered to the approval event types — in any order and
    from any number of sessions (records pair per session; requests are
    seeded before resolutions are applied). Parsing is lenient: approvals
    paused without the routing carrier project with ``routing=None``, and
    unstamped resolutions (made through the raw primitive) project with
    ``outcome=None`` — the audit never raises on foreign approvals. The
    projection records what callers asserted; it does not authenticate the
    stamp — a caller at operator trust level resolving through the raw
    primitive can write any resolution metadata, including this stamp.
    """
    materialized = list(events)
    records: dict[tuple[str, str], dict[str, Any]] = {}
    pending_by_key: dict[tuple[str, str], PendingToolApproval] = {}
    identities: dict[tuple[str, str], ToolRoundIdentity] = {}
    contradictory_request_keys: set[tuple[str, str]] = set()
    resolution_context_descriptors: dict[tuple[str, str], dict[str, Any]] = {}
    call_resolution_states: dict[tuple[str, str, str], str] = {}
    resolution_descriptors: dict[tuple[str, str], dict[str, Any]] = {}
    contradictory_resolution_keys: set[tuple[str, str]] = set()
    for event in materialized:
        if event.type != EventType.TOOL_CALL_APPROVAL_REQUESTED:
            continue
        try:
            pending = PendingToolApproval.from_event(event)
        except (TypeError, ValueError):
            for approval_id in _approval_request_candidate_ids(event):
                record_key = (event.session_id, approval_id)
                contradictory_request_keys.add(record_key)
                records.pop(record_key, None)
                identities.pop(record_key, None)
            continue
        routing: BusinessApprovalRouting | None
        try:
            routing = business_approval_routing(pending)
        except BusinessApprovalRoutingMissing:
            routing = None
        record_key = (event.session_id, pending.approval_id)
        previous_pending = pending_by_key.setdefault(record_key, pending)
        if previous_pending != pending:
            contradictory_request_keys.add(record_key)
            records.pop(record_key, None)
            identities.pop(record_key, None)
            continue
        if record_key in contradictory_request_keys:
            continue
        identities[record_key] = ToolRoundIdentity(
            model_step_id=pending.model_step_id,
            model_attempt_id=pending.model_attempt_id,
            tool_round_id=pending.tool_round_id,
        )
        records[record_key] = {
            "approval_id": pending.approval_id,
            "session_id": event.session_id,
            "tool_call_id": pending.tool_call_id,
            "tool_name": pending.tool_name,
            "agent_name": pending.agent_name,
            "routing": routing,
            "resolution_state": BusinessApprovalResolutionState.PENDING,
            "decision": None,
            "outcome": None,
            "condition_text": None,
            "approver_id": None,
            "approver_tier": None,
            "resolved_by": None,
            "reason": None,
            "expired": False,
            "requested_at": event.timestamp,
            "resolved_at": None,
        }

    for event in materialized:
        if event.type not in {
            EventType.TOOL_CALL_APPROVED,
            EventType.TOOL_CALL_APPROVAL_DENIED,
            EventType.TOOL_CALL_BLOCKED,
        }:
            continue
        approval_id = event.payload.get("approval_id")
        if type(approval_id) is not str:
            continue
        record_key = (event.session_id, approval_id)
        if record_key not in records or not identities[record_key].matches_payload(event.payload):
            continue
        is_ambiguous_block_candidate = event.type is EventType.TOOL_CALL_BLOCKED and (
            "requested_decision" in event.payload
            or event.payload.get("decision") == "ambiguous"
            or event.payload.get("blocked_by") == "policy_evaluation_ambiguous"
        )
        if event.type is EventType.TOOL_CALL_BLOCKED and not is_ambiguous_block_candidate:
            continue
        try:
            resolution_call = pending_tool_call_for_approval_event(
                event=event,
                approval=pending_by_key[record_key],
            )
        except ValueError:
            contradictory_resolution_keys.add(record_key)
            records.pop(record_key, None)
            continue
        is_gating_call = resolution_call.tool_call_id == records[record_key]["tool_call_id"]
        is_valid_ambiguous_block = (
            event.type is EventType.TOOL_CALL_BLOCKED
            and resolution_call.policy_evidence is ToolPolicyEvidence.AMBIGUOUS
            and resolution_call.policy_decision is None
            and event.payload.get("decision") == "ambiguous"
            and event.payload.get("blocked_by") == "policy_evaluation_ambiguous"
            and event.payload.get("requested_decision") == ToolApprovalDecision.APPROVE.value
        )
        if event.type is EventType.TOOL_CALL_BLOCKED and not is_valid_ambiguous_block:
            contradictory_resolution_keys.add(record_key)
            records.pop(record_key, None)
            continue
        if event.type is EventType.TOOL_CALL_APPROVED and not (
            resolution_call.policy_evidence
            in {
                None,
                ToolPolicyEvidence.AUTHORITATIVE,
            }
            and resolution_call.policy_decision == ToolPolicyDecision.REQUIRE_APPROVAL.value
        ):
            contradictory_resolution_keys.add(record_key)
            records.pop(record_key, None)
            continue
        is_ambiguous_block = event.type is EventType.TOOL_CALL_BLOCKED
        call_resolution_key = (*record_key, resolution_call.tool_call_id)
        resolution_state = str(event.type)
        previous_call_resolution_state = call_resolution_states.setdefault(
            call_resolution_key,
            resolution_state,
        )
        if previous_call_resolution_state != resolution_state:
            contradictory_resolution_keys.add(record_key)
            records.pop(record_key, None)
            continue
        normalized_decision = (
            event.payload.get("requested_decision")
            if is_ambiguous_block
            else ToolApprovalDecision.APPROVE.value
            if event.type is EventType.TOOL_CALL_APPROVED
            else ToolApprovalDecision.DENY.value
        )
        resolution_context = copy_durable_json_value(
            {
                "decision": normalized_decision,
                "expired": event.payload.get("expired") is True,
                "resolved_by": event.payload.get("resolved_by"),
                "reason": (
                    event.payload.get("resolution_reason")
                    if is_ambiguous_block
                    else event.payload.get("reason")
                ),
                "metadata": event.payload.get("metadata"),
            },
            "approval_resolution_context_descriptor",
        )
        previous_context = resolution_context_descriptors.setdefault(
            record_key,
            resolution_context,
        )
        if previous_context != resolution_context:
            contradictory_resolution_keys.add(record_key)
            records.pop(record_key, None)
            continue
        if record_key in contradictory_resolution_keys:
            continue
        if not is_gating_call:
            # Runtime resolution events are round-scoped, but only the exact
            # gating call owns the approval decision. Valid sibling outcomes
            # corroborate operator context without resolving the projection.
            continue
        descriptor = copy_durable_json_value(
            {
                "resolution_state": str(event.type),
                **resolution_context,
            },
            "approval_resolution_descriptor",
        )
        previous_descriptor = resolution_descriptors.setdefault(record_key, descriptor)
        if previous_descriptor != descriptor:
            contradictory_resolution_keys.add(record_key)
            records.pop(record_key, None)
            continue
        if record_key in contradictory_resolution_keys:
            continue
        record = records[record_key]

        if is_ambiguous_block:
            record["resolution_state"] = BusinessApprovalResolutionState.BLOCKED
            record["decision"] = None
        elif event.type is EventType.TOOL_CALL_APPROVED:
            record["resolution_state"] = BusinessApprovalResolutionState.APPROVED
            record["decision"] = ToolApprovalDecision.APPROVE
        else:
            record["resolution_state"] = BusinessApprovalResolutionState.DENIED
            record["decision"] = ToolApprovalDecision.DENY
        record["resolved_at"] = event.timestamp
        record["expired"] = event.payload.get("expired") is True
        actor_payload = event.payload.get("resolved_by")
        if type(actor_payload) is dict:
            try:
                record["resolved_by"] = ResolutionActor.model_validate(actor_payload)
            except ValidationError:
                record["resolved_by"] = None
        else:
            record["resolved_by"] = None
        raw_reason = (
            event.payload.get("resolution_reason")
            if is_ambiguous_block
            else event.payload.get("reason")
        )
        record["reason"] = raw_reason if type(raw_reason) is str else None
        stamp = event.payload.get("metadata")
        stamp = (
            stamp.get(BUSINESS_APPROVAL_RESOLUTION_METADATA_KEY) if type(stamp) is dict else None
        )
        if type(stamp) is dict and stamp.get("kind") == BUSINESS_APPROVAL_RESOLUTION_KIND:
            try:
                record["outcome"] = BusinessApprovalOutcome(stamp.get("outcome"))
            except ValueError:
                record["outcome"] = None
            condition_text = stamp.get("condition_text")
            record["condition_text"] = condition_text if type(condition_text) is str else None
            approver_id = stamp.get("approver_id")
            record["approver_id"] = approver_id if type(approver_id) is str else None
            approver_tier = stamp.get("approver_tier")
            record["approver_tier"] = approver_tier if type(approver_tier) is str else None
        else:
            record["outcome"] = None
            record["condition_text"] = None
            record["approver_id"] = None
            record["approver_tier"] = None

    return [BusinessApprovalRecord(**record) for record in records.values()]
