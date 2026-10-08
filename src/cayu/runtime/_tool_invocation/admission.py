"""Targeted tool admission, policy evaluation and durable approval pauses."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timedelta
from typing import Any, Literal
from uuid import uuid4

from cayu._validation import (
    copy_durable_json_object,
    copy_durable_json_value,
    copy_durable_metadata,
    copy_durable_record,
    copy_json_value,
)
from cayu.agents import AgentSpec
from cayu.approvals.tools import (
    PendingToolApproval,
    ToolPolicyEvidence,
    copy_pending_tool_approval,
)
from cayu.budgets._run_limit_accounting import (
    pause_run_limit_accounting_context,
)
from cayu.budgets.base import BudgetLimit, copy_request_budget_limits
from cayu.budgets.run_limits import RunLimits, copy_run_limits
from cayu.context.structured_output import StructuredOutputSpec, copy_structured_output_spec
from cayu.context.thinking import ThinkingConfig
from cayu.events import (
    Event,
    EventType,
)
from cayu.execution_units import ToolRoundIdentity, copy_tool_round_identity
from cayu.mcp.tools import McpToolAdapter
from cayu.messages import Message
from cayu.providers.retry_policy import RetryPolicy, copy_retry_policy
from cayu.runtime import _approval_publication as approval_publication
from cayu.runtime import _approval_support as approval_support
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _tool_execution as tool_execution
from cayu.runtime import _tool_round_recovery as tool_round_recovery
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime._interruption_coordinator import (
    _PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY,
)
from cayu.runtime._invocation_lifecycle import InvocationContext
from cayu.runtime._phase_timing import timed_phase
from cayu.runtime._session_queries import query_all_event_records
from cayu.runtime._tool_invocation.context import (
    _environment_name,
    _published_argument_presence,
    _workspace_id,
)
from cayu.runtime._tool_invocation.hooks import (
    _redact_event_for_invocation,
)
from cayu.runtime._tool_round_staging import (
    _event_with_tool_round_authority,
    _redactor_for_tool_calls,
)
from cayu.runtime.public_authority import parse_public_authority_alias
from cayu.sessions import _pending_approval_reader as pending_approval_reader
from cayu.sessions import _pending_tool_round as pending_rounds
from cayu.sessions import _pending_tool_round_reader as pending_round_reader
from cayu.sessions._checkpoint_secret_validation import (
    require_secret_free_durable_object as _require_secret_free_durable_object,
)
from cayu.sessions.base import (
    Session,
    SessionStatus,
    SessionStore,
    runtime_publication_checkpoint_value_digest,
)
from cayu.sessions.event_queries import EventQuery
from cayu.tools.base import (
    ToolEffect,
)
from cayu.tools.catalogue import CALL_TOOL_NAME
from cayu.tools.commands import ExecCommandTool
from cayu.tools.discovery import (
    TOOL_DISCOVERY_REFERENCE_PREFIX,
    TOOL_DISCOVERY_VIEW_OPERATION_KEY,
    current_tool_discovery_view,
    discovered_tool_rejection_event,
    resolved_discovered_tool_invocation,
    tool_discovery_generation_id,
    tool_discovery_record_matches_descriptor,
    tool_discovery_reference_rejection_reason,
)
from cayu.tools.exposure import (
    tool_capability_ceiling_from_session_metadata,
)
from cayu.tools.gateway import (
    CallToolEnvelope,
    rejected_targeted_tool_invocation,
    resolved_targeted_tool_invocation,
    tool_argument_validation_error,
    unresolved_gateway_rejection_event,
    validate_effective_tool_arguments,
)
from cayu.tools.gateway import arguments_sha256 as targeted_arguments_sha256
from cayu.tools.grants import (
    TARGETED_TOOL_REFERENCE_FIELD_NAME,
    TargetedToolUseDisposition,
    TargetedToolUseRejectionReason,
    TargetedToolUseRequest,
    targeted_tool_use_rejection_event,
    targeted_tool_use_rejection_reason,
    targeted_tool_view_generation_id,
)
from cayu.tools.patches import ApplyPatchTool
from cayu.tools.policy import (
    GuardedToolPolicy,
    TaintAwareToolPolicy,
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
    metadata_with_taint_labels,
    taint_labels_from_metadata,
)
from cayu.tools.structured_commands import RunCommandTool
from cayu.vaults.redaction import SecretRedactor


class ToolApprovalRequired(Exception):
    """Internal control signal for a durably checkpointed approval pause."""

    def __init__(self, approval: PendingToolApproval) -> None:
        super().__init__(f"Tool call requires approval: {approval.tool_name}")
        self.approval = copy_pending_tool_approval(approval)


_INTERRUPTION_TYPE_TOOL_APPROVAL_REQUIRED = "tool_approval_required"


def _require_matching_policy_round(
    *,
    pending_round: pending_rounds.PendingToolRound,
    tool_round_identity: ToolRoundIdentity,
    tool_calls: list[runtime_records.ToolCallRequest],
) -> None:
    if pending_rounds.pending_tool_round_identity(pending_round) != tool_round_identity:
        raise RuntimeError("Pending tool round identity changed before policy publication.")
    expected_calls = [runtime_records.copy_tool_call_request(call) for call in tool_calls]
    pending_calls = tool_round_recovery.pending_round_tool_calls(pending_round)
    if pending_calls != expected_calls:
        raise RuntimeError("Pending tool round calls changed before policy publication.")


def _planned_pending_tool_round(
    *,
    pending_round: pending_rounds.PendingToolRound,
    tool_calls: list[runtime_records.ToolCallRequest],
    policy_outcomes: list[runtime_records.ToolCallPolicyOutcome] | None,
    active_taint_by_id: Mapping[str, frozenset[str]],
    redactor: SecretRedactor,
    deferred_messages: list[Message] | None = None,
) -> pending_rounds.PendingToolRound:
    payload = pending_round.model_dump(mode="json")
    # Checkpoints written before policy-context versioning are valid while the
    # originating run is still evaluating their policy plan. Publishing a
    # freshly evaluated live-run plan, or a fail-closed ambiguous recovery
    # plan, is the authoritative migration boundary. Recovery preserves an
    # unversioned round only when every registered call already carries a
    # recognized durable policy decision.
    payload["policy_context_version"] = 1
    payload["policy_state"] = "planned"
    payload["tool_calls"] = [
        call.model_dump(mode="json")
        for call in approval_support.pending_tool_call_approvals(
            tool_calls=tool_calls,
            policy_outcomes=policy_outcomes,
            active_taint_by_id=active_taint_by_id,
            redactor=redactor,
        )
    ]
    if deferred_messages is not None:
        payload["deferred_messages"] = [
            message.model_dump(mode="json") for message in deferred_messages
        ]
    return pending_rounds.PendingToolRound.model_validate(payload)


def _copy_agent_spec(spec: AgentSpec) -> AgentSpec:
    if type(spec) is not AgentSpec:
        raise TypeError("Agent registration requires an AgentSpec.")
    return AgentSpec(
        name=spec.name,
        model=spec.model,
        provider_name=spec.provider_name,
        system_prompt=spec.system_prompt,
        workflow_tool_names=spec.workflow_tool_names,
        authoring_state=spec.authoring_state,
        metadata=copy_durable_metadata(spec.metadata),
        provider_options=copy_json_value(spec.provider_options, "provider_options"),
        thinking=spec.thinking,
    )


def _tool_effect(
    registered_agent: runtime_records.RegisteredAgentState,
    tool_call: runtime_records.ToolCallRequest,
) -> ToolEffect:
    registered_tool = registered_agent.executable_tool(tool_call.name)
    if registered_tool is None:
        return ToolEffect.EXTERNAL
    return registered_tool.effect


def _taint_policy(policy: ToolPolicy) -> TaintAwareToolPolicy | None:
    while type(policy) is GuardedToolPolicy:
        policy = policy.then
    return policy if isinstance(policy, TaintAwareToolPolicy) else None


def _registered_tool_argument_error(registered_tool, arguments):
    if registered_tool is None:
        return None
    if isinstance(registered_tool.tool, (ExecCommandTool, RunCommandTool, ApplyPatchTool)):
        # These built-ins own content-free command validation before effects.
        # Their schemas are provider hints; replacing the preflight would lose
        # selector/process denial codes and their repair instructions.
        return None
    return tool_argument_validation_error(arguments, registered_tool.schema)


class ToolInvocationAdmission:
    """Admit calls independently of execution, hooks and terminal publication."""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        event_writer: RuntimeEventWriter,
        secret_redactor: SecretRedactor,
        clock: Callable[[], datetime],
    ) -> None:
        self._session_store = session_store
        self._event_writer = event_writer
        self._secret_redactor = secret_redactor
        self._clock = clock

    def _targeted_tool_use_request(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        interaction_id: str,
        task_id: str | None,
        tool_ref: str,
        tool_id: str,
        tool_name: str,
        descriptor_version: str,
        schema_fingerprint: str,
        model_step_id: str,
        outer_tool_call_id: str,
        arguments_digest: str,
        invocation_id: str,
    ) -> TargetedToolUseRequest:
        return TargetedToolUseRequest(
            tool_ref=tool_ref,
            session_id=session.id,
            interaction_id=interaction_id,
            generation_id=targeted_tool_view_generation_id(
                session_id=session.id,
                root_invocation_id=session.invocation.root_invocation_id,
            ),
            agent_name=registered_agent.spec.name,
            task_id=task_id,
            environment_name=_environment_name(registered_environment),
            principal=session.invocation.origin.subject,
            tenant=session.invocation.origin.tenant,
            catalogue_revision=registered_agent.tool_catalogue.revision,
            descriptor_version=descriptor_version,
            schema_fingerprint=schema_fingerprint,
            tool_id=tool_id,
            tool_name=tool_name,
            model_step_id=model_step_id,
            outer_tool_call_id=outer_tool_call_id,
            arguments_sha256=arguments_digest,
            invocation_id=invocation_id,
            expected_run_epoch=session.run_epoch,
        )

    @timed_phase("authorization")
    async def resolve_targeted_calls(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        pending_round: pending_rounds.PendingToolRound,
        invocation_context: InvocationContext | None = None,
    ) -> tuple[
        pending_rounds.PendingToolRound,
        list[runtime_records.ToolCallRequest],
        tuple[Event, ...],
    ]:
        """Resolve and durably bind dynamic-tool calls before policy evaluation."""

        if invocation_context is not None and (
            invocation_context.binding.session_id != session.id
            or registered_agent is not invocation_context.registered_agent
            or registered_environment is not invocation_context.registered_environment
        ):
            raise RuntimeError("Targeted-tool resolution substituted frozen invocation authority.")

        source_calls = tool_round_recovery.pending_round_tool_calls(pending_round)
        if not any(
            call.name == CALL_TOOL_NAME
            or call.targeted_tool_grant_id is not None
            or call.targeted_tool_invocation is not None
            or call.targeted_tool_rejection is not None
            for call in source_calls
        ):
            return pending_round, source_calls, ()
        interaction_id = pending_round.interaction_id
        if interaction_id is None:
            raise RuntimeError("A pending dynamic-tool round has no interaction identity.")
        if pending_round.policy_state != "unplanned" and any(
            (call.name == CALL_TOOL_NAME or call.targeted_tool_grant_id is not None)
            and call.targeted_tool_invocation is None
            and call.targeted_tool_rejection is None
            for call in source_calls
        ):
            raise RuntimeError("An unresolved dynamic-tool call cannot carry a policy plan.")

        records = (
            ()
            if registered_agent.targeted_tool_mode is None
            else await self._session_store.list_targeted_tool_grants(
                session.id,
                interaction_id=interaction_id,
            )
        )
        records_by_ref = {record.tool_ref: record for record in records}
        records_by_id = {record.grant_id: record for record in records}
        if len(records_by_ref) != len(records) or len(records_by_id) != len(records):
            raise RuntimeError("Targeted grant state contains duplicate identities.")
        resolved_calls: list[runtime_records.ToolCallRequest] = []
        resolution_events: list[Event] = []
        identity = pending_rounds.pending_tool_round_identity(pending_round)
        observed_at = self._clock()
        capability_ceiling = tool_capability_ceiling_from_session_metadata(session.metadata)
        ceiling_names = frozenset(capability_ceiling.tool_names)
        discovery_view = None
        if registered_agent.tool_discovery_mode is not None:
            raw_discovery_view = await self._session_store.load_session_operation(
                session.id,
                TOOL_DISCOVERY_VIEW_OPERATION_KEY,
            )
            discovery_view = current_tool_discovery_view(
                raw_discovery_view,
                session_id=session.id,
                generation_id=tool_discovery_generation_id(
                    session_id=session.id,
                    root_invocation_id=session.invocation.root_invocation_id,
                ),
                agent_name=registered_agent.spec.name,
                catalogue=registered_agent.tool_catalogue,
                ceiling=capability_ceiling,
            )
        discovery_records_by_ref = (
            {}
            if discovery_view is None
            else {grant.tool_ref: grant for grant in discovery_view.grants}
        )
        discovery_records_by_id = (
            {}
            if discovery_view is None
            else {grant.grant_id: grant for grant in discovery_view.grants}
        )
        if len(discovery_records_by_id) != len(discovery_records_by_ref):
            raise RuntimeError("Tool discovery view contains duplicate grant identities.")

        async def publish_persisted(event: Event) -> None:
            delivered = await self._event_writer.fan_out_persisted([event])
            resolution_events.extend(delivered)

        async def publish_new(event: Event) -> None:
            resolution_events.append(await self._event_writer.emit(event))

        async def record_for_reference(tool_ref: str):
            record = records_by_ref.get(tool_ref)
            if record is not None:
                return record
            if registered_agent.targeted_tool_mode is None:
                return None
            try:
                parsed = parse_public_authority_alias(tool_ref)
            except (TypeError, ValueError):
                return None
            if parsed is None or parsed.field_name != TARGETED_TOOL_REFERENCE_FIELD_NAME:
                return None
            grant_id = await self._session_store.resolve_public_authority_alias(
                tool_ref,
                field_name=TARGETED_TOOL_REFERENCE_FIELD_NAME,
                scope_session_id=session.id,
            )
            return None if grant_id is None else records_by_id.get(grant_id)

        def invocation_id_for(call: runtime_records.ToolCallRequest) -> str:
            material = (
                f"{session.id}\x00{identity.tool_round_id}\x00{identity.model_step_id}\x00{call.id}"
            ).encode()
            return f"sha256:{hashlib.sha256(material).hexdigest()}"

        def rejected_call(
            call: runtime_records.ToolCallRequest,
            *,
            reason: TargetedToolUseRejectionReason,
            event: Event,
            dispatch_kind: Literal["gateway", "native"] = "gateway",
            model_tool_name: str = CALL_TOOL_NAME,
        ) -> runtime_records.ToolCallRequest:
            return runtime_records.ToolCallRequest(
                id=call.id,
                name=model_tool_name,
                arguments=copy_durable_json_value(call.transcript_arguments, "arguments"),
                targeted_tool_grant_id=call.targeted_tool_grant_id,
                model_tool_name=model_tool_name,
                targeted_tool_rejection=rejected_targeted_tool_invocation(
                    reason=reason,
                    event=event,
                    dispatch_kind=dispatch_kind,
                    model_tool_name=model_tool_name,
                ),
            )

        for call in source_calls:
            if call.targeted_tool_rejection is not None:
                resolved_calls.append(runtime_records.copy_tool_call_request(call))
                continue

            if call.targeted_tool_invocation is not None:
                invocation = call.targeted_tool_invocation
                record = (
                    await record_for_reference(invocation.tool_ref)
                    if invocation.dispatch_kind == "gateway" and invocation.tool_ref is not None
                    else records_by_id.get(invocation.grant_id)
                )
                discovered_record = (
                    discovery_records_by_ref.get(invocation.tool_ref)
                    if invocation.dispatch_kind == "gateway" and invocation.tool_ref is not None
                    else discovery_records_by_id.get(invocation.grant_id)
                )
                if record is None and discovered_record is not None:
                    try:
                        descriptor = registered_agent.tool_catalogue.descriptor_for_id(
                            discovered_record.tool_id
                        )
                    except KeyError:
                        descriptor = None
                    if (
                        discovered_record.grant_id != invocation.grant_id
                        or discovered_record.tool_id != invocation.tool_id
                        or discovered_record.tool_name != invocation.effective_tool_name
                        or discovered_record.catalogue_revision != invocation.catalogue_revision
                        or discovered_record.descriptor_version != invocation.descriptor_version
                        or discovered_record.schema_fingerprint != invocation.schema_fingerprint
                        or descriptor is None
                        or not tool_discovery_record_matches_descriptor(
                            discovered_record,
                            descriptor,
                        )
                        or not _tool_dispatch_authority_is_current(
                            registered_agent,
                            discovered_record.tool_name,
                        )
                        or descriptor.name not in ceiling_names
                        or invocation.session_id != session.id
                        or invocation.interaction_id != interaction_id
                        or invocation.model_step_id != identity.model_step_id
                        or invocation.outer_tool_call_id != call.id
                        or invocation.model_tool_name != call.model_tool_name
                        or (
                            invocation.dispatch_kind == "native"
                            and (
                                invocation.tool_ref is not None
                                or call.model_tool_name != discovered_record.tool_name
                            )
                        )
                        or targeted_arguments_sha256(call.arguments) != invocation.arguments_sha256
                    ):
                        raise RuntimeError(
                            "Resolved discovered-tool checkpoint conflicts with view state."
                        )
                    resolved_calls.append(runtime_records.copy_tool_call_request(call))
                    continue
                if (
                    record is None
                    or record.grant_id != invocation.grant_id
                    or record.tool_id != invocation.tool_id
                    or record.tool_name != invocation.effective_tool_name
                    or record.catalogue_revision != invocation.catalogue_revision
                    or record.descriptor_version != invocation.descriptor_version
                    or record.schema_fingerprint != invocation.schema_fingerprint
                    or invocation.session_id != session.id
                    or invocation.interaction_id != interaction_id
                    or invocation.model_step_id != identity.model_step_id
                    or invocation.outer_tool_call_id != call.id
                    or invocation.model_tool_name != call.model_tool_name
                    or targeted_arguments_sha256(call.arguments) != invocation.arguments_sha256
                ):
                    raise RuntimeError(
                        "Resolved targeted-tool checkpoint conflicts with grant state."
                    )
                if not _tool_dispatch_authority_is_current(
                    registered_agent,
                    invocation.effective_tool_name,
                ):
                    raise RuntimeError(
                        "Resolved targeted-tool checkpoint lost live dispatch authority."
                    )
                request = self._targeted_tool_use_request(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    interaction_id=interaction_id,
                    task_id=pending_round.task_id,
                    tool_ref=record.tool_ref,
                    tool_id=record.tool_id,
                    tool_name=record.tool_name,
                    descriptor_version=record.descriptor_version,
                    schema_fingerprint=record.schema_fingerprint,
                    model_step_id=identity.model_step_id,
                    outer_tool_call_id=call.id,
                    arguments_digest=invocation.arguments_sha256,
                    invocation_id=invocation.invocation_id,
                )
                result = await self._session_store.bind_targeted_tool_grant_use(
                    request,
                    observed_at=observed_at,
                )
                if result.event is None:
                    raise RuntimeError("Targeted tool rejoin returned no event evidence.")
                await publish_persisted(result.event)
                if result.disposition is TargetedToolUseDisposition.REJECTED:
                    if result.reason is None:
                        raise RuntimeError("Targeted tool rejoin rejection lost its reason.")
                    resolved_calls.append(
                        rejected_call(
                            call,
                            reason=result.reason,
                            event=result.event,
                            dispatch_kind=invocation.dispatch_kind,
                            model_tool_name=invocation.model_tool_name,
                        )
                    )
                    continue
                if (
                    result.disposition is not TargetedToolUseDisposition.REJOINED
                    or result.binding is None
                    or result.binding.use_id != invocation.use_id
                ):
                    raise RuntimeError(
                        "Resolved targeted-tool checkpoint did not rejoin its binding."
                    )
                resolved_calls.append(runtime_records.copy_tool_call_request(call))
                continue

            if call.name != CALL_TOOL_NAME and call.targeted_tool_grant_id is None:
                resolved_calls.append(runtime_records.copy_tool_call_request(call))
                continue

            invocation_id = invocation_id_for(call)
            dispatch_kind: Literal["gateway", "native"]
            model_tool_name = call.name
            tool_ref: str | None
            effective_arguments: dict[str, Any]
            discovered_record = None
            if call.name == CALL_TOOL_NAME:
                dispatch_kind = "gateway"
                raw_digest = targeted_arguments_sha256(call.arguments)
                try:
                    envelope = CallToolEnvelope.model_validate(call.arguments)
                except (TypeError, ValueError):
                    event = unresolved_gateway_rejection_event(
                        session_id=session.id,
                        interaction_id=interaction_id,
                        agent_name=registered_agent.spec.name,
                        environment_name=_environment_name(registered_environment),
                        model_step_id=identity.model_step_id,
                        outer_tool_call_id=call.id,
                        arguments_digest=raw_digest,
                        reason=TargetedToolUseRejectionReason.MALFORMED,
                        timestamp=observed_at,
                    )
                    await publish_new(event)
                    resolved_calls.append(
                        rejected_call(
                            call,
                            reason=TargetedToolUseRejectionReason.MALFORMED,
                            event=event,
                        )
                    )
                    continue
                tool_ref = envelope.tool_ref
                effective_arguments = envelope.arguments
                selected_grant_id = call.targeted_tool_grant_id
                if selected_grant_id is None:
                    record = await record_for_reference(envelope.tool_ref)
                    if record is None:
                        discovered_record = discovery_records_by_ref.get(envelope.tool_ref)
                else:
                    record = records_by_id.get(selected_grant_id)
                    resolved_grant_id = await self._session_store.resolve_public_authority_alias(
                        envelope.tool_ref,
                        field_name="tool_ref",
                        scope_session_id=session.id,
                    )
                    if record is None or resolved_grant_id != selected_grant_id:
                        raise RuntimeError(
                            "Runtime-selected call_tool reference conflicts with durable grant "
                            "state."
                        )
            else:
                dispatch_kind = "native"
                selected_grant_id = call.targeted_tool_grant_id
                if selected_grant_id is None:  # pragma: no cover - branch condition invariant
                    raise AssertionError("Native dynamic-tool call lost its grant selection.")
                record = records_by_id.get(selected_grant_id)
                if record is None:
                    discovered_record = discovery_records_by_id.get(selected_grant_id)
                if record is not None and record.tool_name != call.name:
                    raise RuntimeError(
                        "Runtime-selected native tool conflicts with durable grant state."
                    )
                if discovered_record is not None and discovered_record.tool_name != call.name:
                    raise RuntimeError(
                        "Runtime-selected native discovery tool conflicts with durable view state."
                    )
                tool_ref = None
                effective_arguments = copy_durable_json_value(call.arguments, "arguments")
            if discovered_record is not None:
                try:
                    descriptor = registered_agent.tool_catalogue.descriptor_for_id(
                        discovered_record.tool_id
                    )
                except KeyError:
                    descriptor = None
                rejection_reason = None
                if discovered_record.tool_name not in ceiling_names:
                    rejection_reason = TargetedToolUseRejectionReason.OUT_OF_CEILING
                elif (
                    descriptor is None
                    or discovered_record.catalogue_revision
                    != registered_agent.tool_catalogue.revision
                    or not tool_discovery_record_matches_descriptor(
                        discovered_record,
                        descriptor,
                    )
                    or not _tool_dispatch_authority_is_current(
                        registered_agent,
                        discovered_record.tool_name,
                    )
                ):
                    rejection_reason = TargetedToolUseRejectionReason.CATALOGUE_DRIFT
                elif not validate_effective_tool_arguments(
                    effective_arguments,
                    descriptor.input_schema_copy(),
                ):
                    rejection_reason = TargetedToolUseRejectionReason.INVALID_ARGUMENTS
                arguments_digest = targeted_arguments_sha256(effective_arguments)
                if rejection_reason is not None:
                    event = discovered_tool_rejection_event(
                        session_id=session.id,
                        interaction_id=interaction_id,
                        agent_name=registered_agent.spec.name,
                        environment_name=_environment_name(registered_environment),
                        model_step_id=identity.model_step_id,
                        outer_tool_call_id=call.id,
                        arguments_sha256=arguments_digest,
                        reason=rejection_reason,
                        timestamp=observed_at,
                    )
                    await publish_new(event)
                    resolved_calls.append(
                        rejected_call(
                            call,
                            reason=rejection_reason,
                            event=event,
                            dispatch_kind=dispatch_kind,
                            model_tool_name=model_tool_name,
                        )
                    )
                    continue
                resolved_calls.append(
                    runtime_records.ToolCallRequest(
                        id=call.id,
                        name=discovered_record.tool_name,
                        arguments=copy_durable_json_value(effective_arguments, "arguments"),
                        model_tool_name=model_tool_name,
                        targeted_tool_invocation=resolved_discovered_tool_invocation(
                            record=discovered_record,
                            session_id=session.id,
                            interaction_id=interaction_id,
                            model_step_id=identity.model_step_id,
                            outer_tool_call_id=call.id,
                            arguments_sha256=arguments_digest,
                            invocation_id=invocation_id,
                            dispatch_kind=dispatch_kind,
                            model_tool_name=model_tool_name,
                        ),
                    )
                )
                continue
            if record is None:
                if dispatch_kind != "gateway" or tool_ref is None:
                    raise RuntimeError("Native dynamic-tool projection lost its durable grant.")
                if registered_agent.tool_discovery_mode is not None and (
                    registered_agent.targeted_tool_mode is None
                    or tool_ref.startswith(TOOL_DISCOVERY_REFERENCE_PREFIX)
                ):
                    arguments_digest = targeted_arguments_sha256(effective_arguments)
                    rejection_reason = tool_discovery_reference_rejection_reason(tool_ref)
                    event = discovered_tool_rejection_event(
                        session_id=session.id,
                        interaction_id=interaction_id,
                        agent_name=registered_agent.spec.name,
                        environment_name=_environment_name(registered_environment),
                        model_step_id=identity.model_step_id,
                        outer_tool_call_id=call.id,
                        arguments_sha256=arguments_digest,
                        reason=rejection_reason,
                        timestamp=observed_at,
                    )
                    await publish_new(event)
                    resolved_calls.append(
                        rejected_call(
                            call,
                            reason=rejection_reason,
                            event=event,
                        )
                    )
                    continue
                request = self._targeted_tool_use_request(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    interaction_id=interaction_id,
                    task_id=pending_round.task_id,
                    tool_ref=tool_ref,
                    tool_id="cayu:unresolved-targeted-reference",
                    tool_name=CALL_TOOL_NAME,
                    descriptor_version=f"sha256:{'0' * 64}",
                    schema_fingerprint=f"sha256:{'0' * 64}",
                    model_step_id=identity.model_step_id,
                    outer_tool_call_id=call.id,
                    arguments_digest=targeted_arguments_sha256(effective_arguments),
                    invocation_id=invocation_id,
                )
                result = await self._session_store.bind_targeted_tool_grant_use(
                    request,
                    observed_at=observed_at,
                )
                if (
                    result.disposition is not TargetedToolUseDisposition.REJECTED
                    or result.reason is None
                    or result.event is None
                ):
                    raise RuntimeError("Unknown call_tool reference unexpectedly resolved.")
                await publish_persisted(result.event)
                resolved_calls.append(rejected_call(call, reason=result.reason, event=result.event))
                continue

            try:
                descriptor = registered_agent.tool_catalogue.descriptor_for_id(record.tool_id)
            except KeyError:
                descriptor = None
            arguments_digest = targeted_arguments_sha256(effective_arguments)
            request = self._targeted_tool_use_request(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                interaction_id=interaction_id,
                task_id=pending_round.task_id,
                tool_ref=record.tool_ref,
                tool_id=(record.tool_id if descriptor is None else descriptor.tool_id),
                tool_name=(record.tool_name if descriptor is None else descriptor.name),
                descriptor_version=(
                    record.descriptor_version if descriptor is None else descriptor.version
                ),
                schema_fingerprint=(
                    record.schema_fingerprint
                    if descriptor is None
                    else descriptor.schema_fingerprint
                ),
                model_step_id=identity.model_step_id,
                outer_tool_call_id=call.id,
                arguments_digest=arguments_digest,
                invocation_id=invocation_id,
            )
            preflight_reason = targeted_tool_use_rejection_reason(
                record,
                request,
                observed_at=observed_at,
            )
            explicit_rejection = None
            if record.tool_name not in ceiling_names:
                explicit_rejection = TargetedToolUseRejectionReason.OUT_OF_CEILING
            elif descriptor is None or not _tool_dispatch_authority_is_current(
                registered_agent,
                descriptor.name,
            ):
                explicit_rejection = TargetedToolUseRejectionReason.CATALOGUE_DRIFT
            if explicit_rejection is not None:
                event = targeted_tool_use_rejection_event(
                    record,
                    request,
                    reason=explicit_rejection,
                    timestamp=observed_at,
                )
                await publish_new(event)
                resolved_calls.append(
                    rejected_call(
                        call,
                        reason=explicit_rejection,
                        event=event,
                        dispatch_kind=dispatch_kind,
                        model_tool_name=model_tool_name,
                    )
                )
                continue
            if preflight_reason is not None:
                result = await self._session_store.bind_targeted_tool_grant_use(
                    request,
                    observed_at=observed_at,
                )
                if (
                    result.disposition is not TargetedToolUseDisposition.REJECTED
                    or result.reason is None
                    or result.event is None
                ):
                    raise RuntimeError("Rejected targeted-tool preflight unexpectedly bound.")
                await publish_persisted(result.event)
                resolved_calls.append(
                    rejected_call(
                        call,
                        reason=result.reason,
                        event=result.event,
                        dispatch_kind=dispatch_kind,
                        model_tool_name=model_tool_name,
                    )
                )
                continue
            if descriptor is None:  # pragma: no cover - explicit rejection above
                raise AssertionError("Callable targeted grant lost its registered descriptor.")
            if not validate_effective_tool_arguments(
                effective_arguments,
                descriptor.input_schema_copy(),
            ):
                event = targeted_tool_use_rejection_event(
                    record,
                    request,
                    reason=TargetedToolUseRejectionReason.INVALID_ARGUMENTS,
                    timestamp=observed_at,
                )
                await publish_new(event)
                resolved_calls.append(
                    rejected_call(
                        call,
                        reason=TargetedToolUseRejectionReason.INVALID_ARGUMENTS,
                        event=event,
                        dispatch_kind=dispatch_kind,
                        model_tool_name=model_tool_name,
                    )
                )
                continue

            result = await self._session_store.bind_targeted_tool_grant_use(
                request,
                observed_at=observed_at,
            )
            if result.event is None:
                raise RuntimeError("Targeted tool binding returned no event evidence.")
            await publish_persisted(result.event)
            if result.disposition is TargetedToolUseDisposition.REJECTED:
                if result.reason is None:
                    raise RuntimeError("Targeted tool binding rejection lost its reason.")
                resolved_calls.append(
                    rejected_call(
                        call,
                        reason=result.reason,
                        event=result.event,
                        dispatch_kind=dispatch_kind,
                        model_tool_name=model_tool_name,
                    )
                )
                continue
            if result.grant is None or result.binding is None:
                raise RuntimeError("Targeted tool binding returned incomplete authority.")
            resolved_calls.append(
                runtime_records.ToolCallRequest(
                    id=call.id,
                    name=result.grant.tool_name,
                    arguments=copy_durable_json_value(effective_arguments, "arguments"),
                    targeted_tool_grant_id=call.targeted_tool_grant_id,
                    model_tool_name=model_tool_name,
                    targeted_tool_invocation=resolved_targeted_tool_invocation(
                        record=result.grant,
                        binding=result.binding,
                        tool_ref=tool_ref,
                        dispatch_kind=dispatch_kind,
                        model_tool_name=model_tool_name,
                    ),
                )
            )

        if resolved_calls == source_calls:
            return pending_round, resolved_calls, tuple(resolution_events)
        redactor = _redactor_for_tool_calls(
            self._secret_redactor,
            registered_agent=registered_agent,
            tool_calls=resolved_calls,
        )
        payload = pending_round.model_dump(mode="json")
        payload["tool_calls"] = [
            record.model_dump(mode="json")
            for record in approval_support.pending_tool_call_approvals(
                tool_calls=resolved_calls,
                policy_outcomes=None,
                default_policy_evidence=ToolPolicyEvidence.UNPLANNED,
                redactor=redactor,
            )
        ]
        resolved_round = pending_rounds.PendingToolRound.model_validate(payload)
        source_payload = pending_round.model_dump(mode="json")
        resolved_payload = _require_secret_free_durable_object(
            resolved_round.model_dump(mode="json"),
            redactor=redactor,
            field_name="pending_tool_round",
            schema_root=pending_rounds.PENDING_TOOL_ROUND_CHECKPOINT_KEY,
        )

        def publish_resolution(
            current_session: Session,
            current_checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any]:
            if current_session.run_epoch != session.run_epoch:
                raise RuntimeError("Targeted-tool resolution lost its run fence.")
            current = (
                {}
                if current_checkpoint is None
                else copy_durable_json_object(current_checkpoint, "checkpoint")
            )
            if current.get(pending_rounds.PENDING_TOOL_ROUND_CHECKPOINT_KEY) != source_payload:
                raise RuntimeError("Pending tool round changed before targeted-tool resolution.")
            current[pending_rounds.PENDING_TOOL_ROUND_CHECKPOINT_KEY] = resolved_payload
            return copy_durable_json_object(current, "checkpoint")

        await self._session_store.transform_checkpoint(session.id, publish_resolution)
        return resolved_round, resolved_calls, tuple(resolution_events)

    async def rejoin_targeted_call(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        tool_call: runtime_records.ToolCallRequest,
        task_id: str | None,
        invocation_context: InvocationContext | None = None,
    ) -> tuple[Event, ...]:
        """Rejoin a previously bound targeted use before paused execution resumes."""

        if invocation_context is not None and (
            invocation_context.binding.session_id != session.id
            or invocation_context.registered_agent is not registered_agent
            or invocation_context.registered_environment is not registered_environment
        ):
            raise RuntimeError("Targeted-tool rejoin lost frozen invocation authority.")

        invocation = tool_call.targeted_tool_invocation
        if invocation is None:
            return ()
        if (
            invocation.session_id != session.id
            or invocation.effective_tool_name != tool_call.name
            or invocation.model_tool_name != tool_call.model_tool_name
            or invocation.outer_tool_call_id != tool_call.id
            or targeted_arguments_sha256(tool_call.arguments) != invocation.arguments_sha256
        ):
            raise RuntimeError("Paused targeted invocation conflicts with its binding.")
        if not _tool_dispatch_authority_is_current(
            registered_agent,
            invocation.effective_tool_name,
        ):
            raise RuntimeError("Paused targeted invocation lost live dispatch authority.")
        tool_ref = invocation.tool_ref
        if registered_agent.tool_discovery_mode is not None:
            ceiling = tool_capability_ceiling_from_session_metadata(session.metadata)
            discovery_view = current_tool_discovery_view(
                await self._session_store.load_session_operation(
                    session.id,
                    TOOL_DISCOVERY_VIEW_OPERATION_KEY,
                ),
                session_id=session.id,
                generation_id=tool_discovery_generation_id(
                    session_id=session.id,
                    root_invocation_id=session.invocation.root_invocation_id,
                ),
                agent_name=registered_agent.spec.name,
                catalogue=registered_agent.tool_catalogue,
                ceiling=ceiling,
            )
            discovery_record = (
                discovery_view.record_for_reference(tool_ref)
                if tool_ref is not None
                else next(
                    (
                        record
                        for record in discovery_view.grants
                        if record.grant_id == invocation.grant_id
                    ),
                    None,
                )
            )
            if discovery_record is not None:
                try:
                    descriptor = registered_agent.tool_catalogue.descriptor_for_id(
                        discovery_record.tool_id
                    )
                except KeyError:
                    descriptor = None
                if (
                    discovery_record.grant_id != invocation.grant_id
                    or discovery_record.tool_id != invocation.tool_id
                    or discovery_record.tool_name != invocation.effective_tool_name
                    or discovery_record.catalogue_revision != invocation.catalogue_revision
                    or discovery_record.descriptor_version != invocation.descriptor_version
                    or discovery_record.schema_fingerprint != invocation.schema_fingerprint
                    or descriptor is None
                    or not tool_discovery_record_matches_descriptor(
                        discovery_record,
                        descriptor,
                    )
                    or descriptor.name not in ceiling.tool_names
                    or (
                        invocation.dispatch_kind == "gateway"
                        and tool_ref != discovery_record.tool_ref
                    )
                    or (
                        invocation.dispatch_kind == "native"
                        and (
                            tool_ref is not None
                            or invocation.model_tool_name != discovery_record.tool_name
                        )
                    )
                ):
                    raise RuntimeError(
                        "Paused discovered-tool invocation conflicts with its durable view."
                    )
                return ()
            if (
                tool_ref is not None and tool_ref.startswith(TOOL_DISCOVERY_REFERENCE_PREFIX)
            ) or registered_agent.targeted_tool_mode is None:
                raise RuntimeError("Paused discovered-tool invocation lost its durable view grant.")
        if tool_ref is None:
            records = await self._session_store.list_targeted_tool_grants(
                session.id,
                interaction_id=invocation.interaction_id,
            )
            matching = [record for record in records if record.grant_id == invocation.grant_id]
            if len(matching) != 1:
                raise RuntimeError("Paused native invocation lost its durable grant.")
            tool_ref = matching[0].tool_ref
        request = self._targeted_tool_use_request(
            session=session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            interaction_id=invocation.interaction_id,
            task_id=task_id,
            tool_ref=tool_ref,
            tool_id=invocation.tool_id,
            tool_name=invocation.effective_tool_name,
            descriptor_version=invocation.descriptor_version,
            schema_fingerprint=invocation.schema_fingerprint,
            model_step_id=invocation.model_step_id,
            outer_tool_call_id=invocation.outer_tool_call_id,
            arguments_digest=invocation.arguments_sha256,
            invocation_id=invocation.invocation_id,
        )
        result = await self._session_store.bind_targeted_tool_grant_use(
            request,
            observed_at=self._clock(),
        )
        if (
            result.disposition is not TargetedToolUseDisposition.REJOINED
            or result.binding is None
            or result.binding.use_id != invocation.use_id
            or result.event is None
        ):
            raise RuntimeError("Paused targeted invocation did not rejoin its binding.")
        delivered = await self._event_writer.fan_out_persisted([result.event])
        return tuple(delivered)

    @timed_phase("authorization")
    async def authorize(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        tool_call: runtime_records.ToolCallRequest,
        request_metadata: dict[str, Any],
        taint_labels: Iterable[str] | None = None,
    ) -> ToolPolicyResult:
        registered_tool = registered_agent.executable_tool(tool_call.name)
        error = _registered_tool_argument_error(registered_tool, tool_call.arguments)
        if error is not None:
            return ToolPolicyResult(
                decision=ToolPolicyDecision.DENY,
                reason=error,
                metadata={"reason": "invalid_arguments"},
            )
        policy_metadata = request_metadata
        if taint_labels:
            policy_metadata = metadata_with_taint_labels(request_metadata, taint_labels)
        policy_result = await registered_agent.tool_policy.authorize(
            ToolPolicyRequest(
                session=session.model_copy(deep=True),
                agent=_copy_agent_spec(registered_agent.spec),
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                tool_effect=_tool_effect(registered_agent, tool_call),
                arguments=tool_call.arguments,
                environment_name=_environment_name(registered_environment),
                workspace_id=_workspace_id(registered_environment),
                metadata=policy_metadata,
            )
        )
        return tool_execution.validate_tool_policy_result(policy_result)

    async def prior_taint_labels(
        self,
        *,
        session_id: str,
        policy: ToolPolicy,
        request_metadata: dict[str, Any],
    ) -> set[str]:
        labels: set[str] = set(taint_labels_from_metadata(request_metadata))
        session = await self._session_store.load(session_id)
        if session is not None:
            labels.update(taint_labels_from_metadata(session.metadata))
        taint_policy = _taint_policy(policy)
        if taint_policy is None:
            return labels
        for event_type in (EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED):
            records = await query_all_event_records(
                self._session_store,
                EventQuery(
                    session_id=session_id,
                    event_type=event_type,
                    limit=5000,
                ),
            )
            for record in records:
                if record.event.tool_name is not None:
                    labels.update(taint_policy.labels_for_source_tool(record.event.tool_name))
        return labels

    @timed_phase("round_commit", shared=True)
    async def pause_for_approval(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        tool_call: runtime_records.ToolCallRequest,
        tool_calls: list[runtime_records.ToolCallRequest],
        policy_outcomes: list[runtime_records.ToolCallPolicyOutcome] | None,
        active_taint_by_id: Mapping[str, frozenset[str]],
        task_id: str | None,
        policy_result: ToolPolicyResult,
        structured_output: StructuredOutputSpec | None,
        thinking: ThinkingConfig | None,
        max_steps: int | None,
        limits: RunLimits | None,
        budget_limits: tuple[BudgetLimit, ...] | None,
        retry_policy: RetryPolicy | None,
        tool_round_identity: ToolRoundIdentity,
        deferred_messages: list[Message] | None = None,
        recovered: bool = False,
    ) -> tuple[PendingToolApproval, list[Event]]:
        tool_round_identity = copy_tool_round_identity(tool_round_identity)
        redactor = _redactor_for_tool_calls(
            self._secret_redactor,
            registered_agent=registered_agent,
            tool_calls=tool_calls,
        )
        checkpoint = await self._session_store.load_checkpoint(session.id)
        checkpoint = {} if checkpoint is None else copy_durable_record(checkpoint, "checkpoint")
        pending_round = pending_round_reader.pending_tool_round_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
            runtime_session=session,
        )
        if pending_round is None:
            raise RuntimeError("Session has no pending tool round for its approval.")
        _require_matching_policy_round(
            pending_round=pending_round,
            tool_round_identity=tool_round_identity,
            tool_calls=tool_calls,
        )
        if (
            pending_approval_reader.pending_approval_from_checkpoint(
                checkpoint,
                redactor=self._secret_redactor,
                consume_on_rejection=True,
            )
            is not None
        ):
            raise RuntimeError("Session already has a pending tool approval.")

        round_ttls = [policy_result.approval_expires_in_seconds]
        if policy_outcomes is not None:
            round_ttls.extend(
                outcome.result.approval_expires_in_seconds
                for outcome in policy_outcomes
                if outcome.result is not None
                and outcome.result.decision == ToolPolicyDecision.REQUIRE_APPROVAL
            )
        bounded_ttls = [ttl for ttl in round_ttls if ttl is not None]
        expires_at: datetime | None = None
        if bounded_ttls:
            expires_at = self._clock() + timedelta(seconds=min(bounded_ttls))
        approval = PendingToolApproval(
            approval_id=str(uuid4()),
            **tool_round_identity.payload(),
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            arguments=copy_json_value(tool_call.arguments, "arguments"),
            agent_name=registered_agent.spec.name,
            environment_name=_environment_name(registered_environment),
            workspace_id=_workspace_id(registered_environment),
            task_id=task_id,
            execution_profile_fingerprint=pending_round.execution_profile_fingerprint,
            publish_arguments=_tool_round_publishes_arguments(
                registered_agent,
                tool_calls,
            ),
            secret_resolution_scope=(
                pending_approval_reader.tool_round_secret_resolution_scope(pending_round)
            ),
            reason=(
                None if policy_result.reason is None else redactor.redact_text(policy_result.reason)
            ),
            metadata=copy_durable_metadata(
                redactor.redact_json_values(policy_result.metadata),
            ),
            tool_calls=approval_support.pending_tool_call_approvals(
                tool_calls=tool_calls,
                policy_outcomes=policy_outcomes,
                active_taint_by_id=active_taint_by_id,
                redactor=redactor,
            ),
            structured_output=copy_structured_output_spec(structured_output),
            thinking=thinking,
            max_steps=max_steps,
            limits=copy_run_limits(limits) if limits is not None else None,
            run_limit_accounting=pause_run_limit_accounting_context(
                pending_round.run_limit_accounting, now=self._clock()
            ),
            budget_limits=(
                copy_request_budget_limits(budget_limits) if budget_limits is not None else None
            ),
            retry_policy=copy_retry_policy(retry_policy) if retry_policy is not None else None,
            expires_at=expires_at,
        )
        approval_payload = _require_secret_free_durable_object(
            approval.model_dump(mode="json"),
            redactor=redactor,
            field_name="pending_tool_approval",
            schema_root=pending_approval_reader.PENDING_TOOL_APPROVAL_CHECKPOINT_KEY,
        )
        planned_round = _planned_pending_tool_round(
            pending_round=pending_round,
            tool_calls=tool_calls,
            policy_outcomes=policy_outcomes,
            active_taint_by_id=active_taint_by_id,
            redactor=redactor,
            deferred_messages=deferred_messages,
        )
        planned_round_payload = _require_secret_free_durable_object(
            planned_round.model_dump(mode="json"),
            redactor=redactor,
            field_name="pending_tool_round",
            schema_root=pending_rounds.PENDING_TOOL_ROUND_CHECKPOINT_KEY,
        )
        # Retain the exact serialized source for the compare-and-swap. Parsing
        # an older checkpoint fills model defaults (including policy version
        # fields), which is useful for behavior but must not change the value
        # we require the atomic publication to replace.
        source_round_payload = copy_json_value(
            checkpoint[pending_rounds.PENDING_TOOL_ROUND_CHECKPOINT_KEY],
            "pending_tool_round",
        )
        # Recovery can own a fenced session in a non-running status (for
        # example an incomplete-session claim over INTERRUPTED). Bind the
        # publication to the exact claimed status and epoch rather than
        # broadening recovery to a fixed set of statuses.
        eligible_statuses = {session.status} if recovered else {SessionStatus.RUNNING}

        def publish_policy_and_approval(
            current_session: Session,
            current_checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any]:
            if (
                current_session.status not in eligible_statuses
                or current_session.run_epoch != session.run_epoch
            ):
                raise RuntimeError("Tool approval publication lost its run fence.")
            current = (
                {}
                if current_checkpoint is None
                else copy_durable_record(current_checkpoint, "checkpoint")
            )
            if (
                current.get(pending_rounds.PENDING_TOOL_ROUND_CHECKPOINT_KEY)
                != source_round_payload
            ):
                raise RuntimeError("Pending tool round changed before approval publication.")
            if pending_approval_reader.PENDING_TOOL_APPROVAL_CHECKPOINT_KEY in current:
                raise RuntimeError("Session already has a pending tool approval.")
            if pending_approval_reader.APPROVAL_RESOLUTION_INTENT_CHECKPOINT_KEY in current:
                raise RuntimeError("Session has an orphaned approval resolution intent.")
            current[pending_rounds.PENDING_TOOL_ROUND_CHECKPOINT_KEY] = planned_round_payload
            current[pending_approval_reader.PENDING_TOOL_APPROVAL_CHECKPOINT_KEY] = approval_payload
            if recovered and _PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY in current:
                interrupt_payload = current[_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY]
                if type(interrupt_payload) is not dict:
                    raise RuntimeError(
                        "Recovered approval has an invalid pending interruption payload."
                    )
                interrupt_payload = copy_json_value(
                    interrupt_payload,
                    "pending_session_interrupt",
                )
                interrupt_payload.update(
                    {
                        "interruption_type": (_INTERRUPTION_TYPE_TOOL_APPROVAL_REQUIRED),
                        **tool_round_identity.payload(),
                        "approval_id": approval.approval_id,
                        "tool_call_id": approval.tool_call_id,
                        **approval_support.bounded_pending_approval_event_payload(
                            approval,
                            redactor=redactor,
                        ),
                        "recovered": True,
                    }
                )
                current[_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY] = interrupt_payload
            # Existing checkpoint roots were validated by their owning
            # publication boundary. Re-scanning the whole checkpoint with this
            # invocation's secrets would misclassify unrelated protocol values
            # (for example a finish reason containing "tool") as leaked data.
            return copy_durable_json_object(current, "checkpoint")

        checkpoint_event = _redact_event_for_invocation(
            _event_with_tool_round_authority(
                Event(
                    type=EventType.SESSION_CHECKPOINTED,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=_environment_name(registered_environment),
                    payload={
                        "checkpoint": pending_approval_reader.PENDING_TOOL_APPROVAL_CHECKPOINT_KEY,
                        "approval_id": approval.approval_id,
                        "tool_call_id": approval.tool_call_id,
                        **tool_round_identity.payload(),
                    },
                ),
                tool_round_identity,
                "approval_id",
            ),
            redactor=redactor,
        )
        requested_payload = {
            **_published_argument_presence(
                tool_call, registered_agent.executable_tool(tool_call.name)
            ),
            **tool_round_identity.payload(),
            "approval_id": approval.approval_id,
            "tool_call_id": approval.tool_call_id,
            **approval_support.bounded_pending_approval_event_payload(
                approval,
                redactor=redactor,
            ),
        }
        if recovered:
            requested_payload["recovered"] = True
        requested_event = _redact_event_for_invocation(
            approval_support.event_with_pending_approval_authority(
                _event_with_tool_round_authority(
                    Event(
                        type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
                        session_id=session.id,
                        agent_name=registered_agent.spec.name,
                        environment_name=_environment_name(registered_environment),
                        tool_name=approval.tool_name,
                        payload=requested_payload,
                    ),
                    tool_round_identity,
                    "approval_id",
                ),
                approval,
            ),
            redactor=redactor,
        )
        events = [checkpoint_event, requested_event]
        target_checkpoint = publish_policy_and_approval(session, checkpoint)
        transcript_cursor = await self._session_store.load_transcript_cursor(session.id)
        prepared = approval_publication.prepare_approval_publication(
            session_id=session.id,
            publication_id=f"approval-open:{approval.approval_id}",
            kind="approval-open",
            intent={
                "schema_version": 1,
                "approval_id": approval.approval_id,
                "tool_call_id": approval.tool_call_id,
                **tool_round_identity.payload(),
                "tool_call_ids": [call.tool_call_id for call in planned_round.tool_calls],
                "source_round_digest": runtime_publication_checkpoint_value_digest(
                    source_round_payload
                ),
                "planned_round_digest": runtime_publication_checkpoint_value_digest(
                    planned_round_payload
                ),
                "approval_digest": runtime_publication_checkpoint_value_digest(approval_payload),
                "event_ids": [event.id for event in events],
            },
            source_checkpoint=checkpoint,
            target_checkpoint=target_checkpoint,
            events=events,
            expected_statuses=eligible_statuses,
            expected_run_epoch=session.run_epoch,
            expected_transcript_cursor=transcript_cursor,
        )
        events = list(prepared.request.events)
        cancellation = await approval_publication.publish_approval_with_exact_replay(
            prepared,
            session_store=self._session_store,
            event_writer=self._event_writer,
        )
        if cancellation is not None:
            raise cancellation
        return approval, events


def _registered_mcp_tool_authority_is_unavailable(
    registered_tool: runtime_records.RegisteredTool | None,
) -> bool:
    if registered_tool is None or not isinstance(registered_tool.tool, McpToolAdapter):
        return False
    return not registered_tool.tool._dispatch_authority_is_current()


def _tool_dispatch_authority_is_current(
    registered_agent: runtime_records.RegisteredAgentState,
    tool_name: str,
) -> bool:
    return not _registered_mcp_tool_authority_is_unavailable(
        registered_agent.executable_tool(tool_name)
    )


def _tool_round_publishes_arguments(
    registered_agent: runtime_records.RegisteredAgentState,
    tool_calls: list[runtime_records.ToolCallRequest],
) -> bool:
    """Require every paused call to permit one shared argument projection."""

    for tool_call in tool_calls:
        registered_tool = registered_agent.executable_tool(tool_call.name)
        if registered_tool is None or not registered_tool.publish_arguments:
            return False
    return True
