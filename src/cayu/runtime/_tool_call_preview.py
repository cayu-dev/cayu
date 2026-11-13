"""What a tool-call proposal would do, decided without recording anything.

``CayuApp.preview_tool_call`` runs the checks a proposal's call meets before it
runs: the tool is registered, the call is inside the capability ceiling and
exposed (or the tool is proposal-only), its arguments match the schema, and the
agent's tool policy decides. The policy sees ``metadata["cayu:dry_run"] = True``
so a stateful policy (a rate limit, an audit sink) can skip counting the call;
only this path sets it (admission drops any value request metadata carries).
No session, event or approval is created and no tool runs.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from cayu.approvals.proposals import ToolCallPreview
from cayu.runtime._proposal_only_tools import model_facing_policy_input
from cayu.runtime._tool_call_proposals import (
    TOOL_CALL_PROPOSAL_MODEL,
    TOOL_CALL_PROPOSAL_PROVIDER_NAME,
)
from cayu.runtime._tool_invocation.admission import _registered_tool_argument_error
from cayu.sessions._tool_call_proposal_record import (
    TOOL_CALL_PROPOSAL_METADATA_KEY,
    tool_call_proposal_record,
)
from cayu.sessions.invocation import (
    InvocationOrigin,
    InvocationOriginClaim,
    InvocationOriginTrust,
    SessionExecutionSource,
    SessionInvocation,
)
from cayu.sessions.records import Session, SessionStatus
from cayu.tools.exposure import (
    AllRegisteredToolsExposurePolicy,
    ToolCapabilityCeiling,
    ToolExposurePolicyRequest,
    resolve_tool_capability_ceiling,
    resolve_tool_exposure,
)
from cayu.tools.policy import ToolPolicyDecision

if TYPE_CHECKING:
    from cayu.applications import CayuApp


def _preview_session(
    agent_name: str,
    *,
    session_id: str,
    origin: InvocationOriginClaim | None,
    labels: Mapping[str, str],
    metadata: Mapping[str, Any],
) -> Session:
    return Session(
        id=session_id,
        agent_name=agent_name,
        provider_name=TOOL_CALL_PROPOSAL_PROVIDER_NAME,
        model=TOOL_CALL_PROPOSAL_MODEL,
        status=SessionStatus.RUNNING,
        causal_budget_id=session_id,
        labels=dict(labels),
        metadata=dict(metadata),
        invocation=SessionInvocation(
            origin=(
                InvocationOrigin(trust=InvocationOriginTrust.UNATTRIBUTED)
                if origin is None
                else InvocationOrigin(
                    trust=InvocationOriginTrust.HOST_ASSERTED,
                    subject=origin.subject,
                    tenant=origin.tenant,
                )
            ),
            root_invocation_id=str(uuid4()),
            root_session_id=session_id,
            source=SessionExecutionSource.SDK_RUN,
        ),
    )


def _registered_environment(app: CayuApp, name: object) -> Any:
    """The agent's default environment registration, factory-backed or not; None without one.

    A factory-backed registration isn't allocated here, but the policy still sees the
    environment name the real proposal's session would run in.
    """

    if name is not None and type(name) is not str:
        return None  # The agent runs without an environment.
    return app._get_registered_environment(name)


async def preview_tool_call(
    app: CayuApp,
    agent_name: str,
    tool_name: str,
    arguments: Mapping[str, Any],
    *,
    tool_call_id: str,
    session_id: str | None,
    invocation_origin: InvocationOriginClaim | None,
    tool_capability_ceiling: ToolCapabilityCeiling | None,
    labels: Mapping[str, str],
    metadata: Mapping[str, Any],
) -> ToolCallPreview:
    registered_agent = app._get_registered_agent(agent_name)
    registered_tool = registered_agent.executable_tool(tool_name)
    proposal_only = tool_name in registered_agent.proposal_tool_names
    fields: dict[str, Any] = {
        "agent_name": agent_name,
        "tool_name": tool_name,
        "proposal_only": proposal_only,
        "effect": None if registered_tool is None else registered_tool.effect.value,
    }
    if registered_tool is None:
        return ToolCallPreview(
            **fields,
            decision=ToolPolicyDecision.DENY,
            blocked_by="unregistered",
            reason=f"{tool_name} is not registered on {agent_name}.",
        )
    ceiling = resolve_tool_capability_ceiling(
        tool_capability_ceiling, registered_agent.tool_capabilities
    )
    preview_id = session_id or f"preview-{uuid4().hex}"
    if tool_name not in ceiling.tool_names:
        exposed = False
    elif proposal_only:
        # The proposal step admits it through the session's runtime record.
        exposed = True
    elif type(registered_agent.tool_exposure_policy) is AllRegisteredToolsExposurePolicy:
        exposed = True
    else:
        registered_tools, capability_ceiling = model_facing_policy_input(
            registered_agent, ceiling.tool_names
        )
        exposure = resolve_tool_exposure(
            registered_agent.tool_exposure_policy,
            ToolExposurePolicyRequest(
                session_id=preview_id,
                agent_name=agent_name,
                provider_name=TOOL_CALL_PROPOSAL_PROVIDER_NAME,
                model=TOOL_CALL_PROPOSAL_MODEL,
                step=1,
                catalogue_revision=registered_agent.tool_catalogue.revision,
                registered_tools=registered_tools,
                capability_ceiling=capability_ceiling,
                metadata=dict(metadata),
            ),
        )
        exposed = tool_name in exposure.tool_names
    if not exposed:
        return ToolCallPreview(
            **fields,
            decision=ToolPolicyDecision.DENY,
            blocked_by="tool_exposure",
            reason=f"{tool_name} is not exposed to {agent_name} for this call.",
        )
    error = _registered_tool_argument_error(registered_tool, dict(arguments))
    if error is not None:
        return ToolCallPreview(
            **fields,
            decision=ToolPolicyDecision.DENY,
            blocked_by="invalid_arguments",
            reason=error,
        )
    from cayu.runtime import _runtime_records as runtime_records

    session = _preview_session(
        agent_name,
        session_id=preview_id,
        origin=invocation_origin,
        labels=labels,
        metadata={
            **metadata,
            TOOL_CALL_PROPOSAL_METADATA_KEY: tool_call_proposal_record(
                tool_name=tool_name, tool_call_id=tool_call_id, arguments=arguments
            ),
        },
    )
    result = await app._tool_round_executor.invocation.admission.authorize(
        session=session,
        registered_agent=registered_agent,
        registered_environment=_registered_environment(
            app, registered_agent.default_environment_name
        ),
        tool_call=runtime_records.ToolCallRequest(
            id=tool_call_id, name=tool_name, arguments=dict(arguments)
        ),
        request_metadata=dict(metadata),
        dry_run=True,
    )
    return ToolCallPreview(
        **fields,
        decision=result.decision,
        blocked_by="tool_policy" if result.decision is ToolPolicyDecision.DENY else None,
        reason=app.redact_json(result.reason),
        metadata=app.redact_json(result.metadata),
    )
