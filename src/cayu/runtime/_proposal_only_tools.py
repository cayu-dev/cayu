"""Keep proposal-only tools away from the model and admit them only as proposed.

An agent registers a tool as proposal-only with ``register_agent(...,
proposal_tools=[...])``. The tool is registered, catalogued and inside the
capability ceiling like any other, but:

- no exposure snapshot a model sees ever contains it, whatever the exposure
  policy, and an exposure policy that names it fails like one naming an
  unregistered tool;
- the one model step that may raise it is a proposal session's step served by
  the runtime's own ``ToolCallProposalProvider``. That step's frozen exposure
  gains the tool only while the session's runtime-owned proposal record names
  it, the record has not been used, and the tool is inside the session's
  ceiling;
- admission then requires the call's name, ID and argument digest to equal the
  record. Anything else is blocked before any tool policy runs.

Tool policy, guards, hooks, approval, secret checks and recovery then apply as
they do to any call.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from cayu._validation import thaw_json_value
from cayu.events import EventType
from cayu.runtime._tool_call_proposals import ToolCallProposalProvider
from cayu.sessions._tool_call_proposal_record import (
    call_matches_tool_call_proposal,
    session_tool_call_proposal,
)
from cayu.sessions.event_queries import EventQuery
from cayu.tools.exposure import (
    NOT_EXPOSED_IN_REQUEST_REASON,
    RegisteredToolCapability,
    ResolvedToolExposure,
    StaticToolExposurePolicy,
    ToolExposurePolicy,
)
from cayu.tools.policy import ToolPolicyDecision, ToolPolicyResult

if TYPE_CHECKING:
    from cayu.providers.base import ModelProvider
    from cayu.runtime import _runtime_records as runtime_records
    from cayu.sessions.records import Session
    from cayu.sessions.store import SessionStore

# Every event that shows a call was raised in an earlier tool round.
_RAISED_CALL_EVENT_TYPES = (
    EventType.TOOL_CALL_APPROVAL_REQUESTED,
    EventType.TOOL_CALL_APPROVED,
    EventType.TOOL_CALL_APPROVAL_DENIED,
    EventType.TOOL_CALL_APPROVAL_EXPIRED,
    EventType.TOOL_CALL_STARTED,
    EventType.TOOL_CALL_COMPLETED,
    EventType.TOOL_CALL_FAILED,
    EventType.TOOL_CALL_BLOCKED,
)


def _rebuilt(
    exposure: ResolvedToolExposure,
    tools: Iterable[RegisteredToolCapability],
    *,
    registered_count: int,
    ceiling_count: int,
) -> ResolvedToolExposure:
    return ResolvedToolExposure(
        profile_id=exposure.profile_id,
        catalogue_revision=exposure.catalogue_revision,
        tools=tuple(tools),
        registered_count=registered_count,
        ceiling_count=ceiling_count,
        metadata=thaw_json_value(exposure.metadata),
    )


def model_facing_exposure(
    registered_agent: runtime_records.RegisteredAgentState,
    exposure: ResolvedToolExposure,
) -> ResolvedToolExposure:
    """Return ``exposure`` without the agent's proposal-only tools."""

    hidden = registered_agent.proposal_tool_names
    if not hidden.intersection(exposure.tool_names):
        return exposure
    return _rebuilt(
        exposure,
        (tool for tool in exposure.tools if tool.name not in hidden),
        registered_count=exposure.registered_count,
        ceiling_count=exposure.ceiling_count,
    )


def model_facing_policy_input(
    registered_agent: runtime_records.RegisteredAgentState,
    ceiling_names: tuple[str, ...],
) -> tuple[tuple[RegisteredToolCapability, ...], tuple[str, ...]]:
    """The registered tools and ceiling an exposure policy may choose from."""

    hidden = registered_agent.proposal_tool_names
    if not hidden:
        return registered_agent.tool_capabilities, ceiling_names
    named = exposure_policy_names_proposal_tool(registered_agent.tool_exposure_policy, hidden)
    if named:
        raise ValueError(
            "Tool exposure selected a proposal-only tool, which is never offered to the "
            f"model: {', '.join(named)}."
        )
    return (
        tuple(tool for tool in registered_agent.tool_capabilities if tool.name not in hidden),
        tuple(name for name in ceiling_names if name not in hidden),
    )


def with_registration_counts(
    registered_agent: runtime_records.RegisteredAgentState,
    exposure: ResolvedToolExposure,
    ceiling_names: tuple[str, ...],
) -> ResolvedToolExposure:
    """Restore the full registered and ceiling counts after a model-facing selection."""

    registered_count = len(registered_agent.tool_capabilities)
    if exposure.registered_count == registered_count:
        return exposure
    return _rebuilt(
        exposure,
        exposure.tools,
        registered_count=registered_count,
        ceiling_count=len(ceiling_names),
    )


async def exposure_with_proposed_tool(
    registered_agent: runtime_records.RegisteredAgentState,
    exposure: ResolvedToolExposure,
    *,
    session: Session,
    provider: ModelProvider,
    ceiling_names: tuple[str, ...],
    session_store: SessionStore,
) -> ResolvedToolExposure:
    """Add the recorded proposal-only tool to a proposal step's exposure, once.

    The tool is added only for a step the runtime's proposal provider serves,
    while the session's runtime-owned record names a proposal-only tool inside
    the ceiling, and before any earlier tool round raised that tool in the
    session. A follow-up envelope sent into a finished proposal session
    therefore finds the tool unexposed.
    """

    if not registered_agent.proposal_tool_names or type(provider) is not ToolCallProposalProvider:
        return exposure
    record = session_tool_call_proposal(session.metadata)
    if record is None:
        return exposure
    tool_name = record["tool_name"]
    if (
        tool_name not in registered_agent.proposal_tool_names
        or tool_name not in ceiling_names
        or tool_name in exposure.tool_names
    ):
        return exposure
    raised = await session_store.query_events(
        EventQuery(
            session_id=session.id,
            event_types=_RAISED_CALL_EVENT_TYPES,
            tool_name=tool_name,
            limit=1,
        )
    )
    if raised:
        return exposure
    names = {*exposure.tool_names, tool_name}
    return _rebuilt(
        exposure,
        (tool for tool in registered_agent.tool_capabilities if tool.name in names),
        registered_count=exposure.registered_count,
        ceiling_count=exposure.ceiling_count,
    )


def proposal_call_refusal(
    registered_agent: runtime_records.RegisteredAgentState,
    session: Session,
    tool_call: runtime_records.ToolCallRequest,
    *,
    frozen_exposure: bool,
) -> ToolPolicyResult | None:
    """Block a proposal-only call that is not the session's recorded proposal.

    It is reached only for a call its round's exposure admitted, which for a
    proposal-only tool means a proposal step (see ``exposure_with_proposed_tool``).
    A round without a frozen exposure snapshot cannot show that, so it is
    blocked. Returns None for every other call. The block is decided before
    any tool policy runs, and carries the same reason as an unexposed call.
    """

    if tool_call.name not in registered_agent.proposal_tool_names:
        return None
    record = session_tool_call_proposal(session.metadata)
    if (
        frozen_exposure
        and record is not None
        and tool_call.targeted_tool_invocation is None
        and call_matches_tool_call_proposal(
            record,
            tool_name=tool_call.name,
            tool_call_id=tool_call.id,
            arguments=tool_call.arguments,
        )
    ):
        return None
    return ToolPolicyResult(
        decision=ToolPolicyDecision.DENY,
        reason=f"{tool_call.name} is proposal-only and this call is not its recorded proposal.",
        metadata={"reason": NOT_EXPOSED_IN_REQUEST_REASON},
    )


def exposure_policy_names_proposal_tool(
    policy: ToolExposurePolicy, proposal_tool_names: frozenset[str]
) -> tuple[str, ...]:
    """Proposal-only tools a statically readable exposure policy names."""

    if type(policy) is not StaticToolExposurePolicy:
        return ()
    return tuple(sorted(proposal_tool_names.intersection(policy.tools)))
