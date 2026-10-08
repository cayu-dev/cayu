"""Shared single-tool invocation, from admission through terminal delivery.

Ordinary rounds and recovery compose this owner from the same admission, hooks,
resources and terminal publisher. Round scheduling and continuation orchestration
remain with their callers; low-level execution remains independently usable.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from copy import deepcopy
from dataclasses import replace
from typing import Any, Literal, cast

from cayu._exception_groups import (
    exception_cause,
    iter_exception_tree,
    set_exception_cause,
)
from cayu._task_wait import (
    await_shielded_task_outcome,
    consume_pending_task_cancellation,
)
from cayu._validation import (
    copy_json_value,
)
from cayu.approvals.tools import (
    ToolPolicyEvidence,
)
from cayu.budgets.base import (
    BudgetLimit,
)
from cayu.events import (
    Event,
    EventType,
    event_with_runtime_payload_authority,
)
from cayu.execution_profiles import (
    EXECUTION_PROFILE_FINGERPRINT_FIELD,
    ExecutionProfileIdentity,
    event_with_execution_profile_authority,
)
from cayu.execution_units import (
    ToolRoundIdentity,
    copy_tool_round_identity,
)
from cayu.mcp.tools import McpToolAdapter
from cayu.runners._cleanup import (
    attach_runner_cancellation_failure,
    runner_cancellation_failure,
)
from cayu.runtime import _approval_support as approval_support
from cayu.runtime import _invocation_secrets as invocation_secrets
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _tool_execution as tool_execution
from cayu.runtime._auxiliary_inference import AuxiliaryInferenceOwner
from cayu.runtime._auxiliary_invocation import AuxiliaryInvocationPolicy
from cayu.runtime._durable_tool_round import (
    _interrupted_tool_call_event,
    _interrupted_tool_call_outcome,
)
from cayu.runtime._environment_exposure import (
    require_environment_exposed,
)
from cayu.runtime._event_writer import RuntimeEventWriter, prepare_runtime_event
from cayu.runtime._invocation_lifecycle import InvocationContext
from cayu.runtime._phase_timing import timed_phase
from cayu.runtime._run_limits import (
    SessionUsageTracker,
)
from cayu.runtime._session_control import (
    SessionControl,
    SessionInterruptedByRequest,
    clear_current_task_cancellation,
)
from cayu.runtime._tool_effect_state import (
    ToolEffectReconciliationCleanupFailure,
    ToolEffectReconciliationRequired,
    ToolEffectStateOwner,
)
from cayu.runtime._tool_invocation.admission import (
    ToolApprovalRequired,
    ToolInvocationAdmission,
    _registered_mcp_tool_authority_is_unavailable,
    _registered_tool_argument_error,
    policy_denial_payload_fields,
)
from cayu.runtime._tool_invocation.cancellation import (
    _await_post_tool_operation,
    _contains_process_signal,
    _grouped_cancellation_evidence,
    _iterate_post_tool_events,
    _raise_preserved_post_tool_cancellation,
    _raise_restored_post_tool_cancellation,
    _receive_restored_post_tool_cancellation,
    retain_cancellation_context,
)
from cayu.runtime._tool_invocation.context import (
    _artifact_store_id,
    _environment_name,
    _event_with_targeted_tool_invocation_authority,
    _published_argument_presence,
    _restore_targeted_tool_invocation_event_authority,
    _targeted_tool_invocation_payload,
    _workspace,
    _workspace_id,
)
from cayu.runtime._tool_invocation.dispatch import ToolInvocationDispatch
from cayu.runtime._tool_invocation.evidence import InvocationEvidence, InvocationPublication
from cayu.runtime._tool_invocation.hooks import (
    ToolInvocationHooks,
    _BeforeToolCallResolution,
    _private_argument_short_circuit_result,
    _project_tool_call_for_hook,
)
from cayu.runtime._tool_invocation.resources import ToolInvocationCall, ToolInvocationResources
from cayu.runtime._tool_invocation.terminal import (
    DeferredTerminalCaptureRecorder,
    DeferredTerminalStager,
    ToolTerminalPublisher,
)
from cayu.runtime._tool_invocation.workspace_capture import (
    _MAX_RETAINED_WORKSPACE_CAPTURE_OPERATIONS,
    _observe_workspace_revision,
    _record_workspace_mutation_after,
    _workspace_binding_generation_id,
    _workspace_mutation_incomplete_event,
    _workspace_mutation_window_id,
    _workspace_observation_terminal_view,
    _workspace_revision_observer_is_runtime_owned,
    _workspace_revision_observer_name,
    _workspace_writer_isolation,
    _WorkspaceCaptureResult,
)
from cayu.runtime._tool_round_staging import (
    _event_with_tool_round_authority,
    _prepare_tool_result_event,
    _redactor_for_tool_calls,
)
from cayu.sessions.base import (
    SessionStore,
)
from cayu.sessions.records import Session
from cayu.tools import _argument_publication as tool_argument_publication
from cayu.tools._operation_boundary import (
    BoundedInvocationOperationRegistry,
    await_invocation_operation,
)
from cayu.tools._redaction import InvocationRedactorSnapshot
from cayu.tools._resources import (
    WorkspaceMutationSettlementError,
)
from cayu.tools._runner import (
    is_current_runner_cancellation_group,
    sanitize_runner_failure,
)
from cayu.tools.base import (
    _TOOL_POLICY_DENIAL_SOURCE,
    ToolResult,
)
from cayu.tools.discovery import (
    ToolDiscoveryProjectionKind,
    resolve_tool_discovery_projection,
)
from cayu.tools.exposure import (
    NOT_EXPOSED_IN_REQUEST_REASON,
    ResolvedToolExposureAuthority,
    unexposed_tool_result,
    validate_resolved_tool_exposure_authority,
)
from cayu.tools.gateway import (
    dynamic_tool_reference_rejection,
    targeted_tool_rejection_content,
)
from cayu.tools.policy import (
    TAINT_LABELS_METADATA_KEY,
    TOOL_POLICY_REAUTHORIZATION_METADATA_KEY,
    ToolPolicyDecision,
    ToolPolicyResult,
)
from cayu.vaults.redaction import SecretRedactor
from cayu.workspaces.checkpoint_lifecycle import (
    begin_workspace_checkpoint_mutation,
    complete_workspace_checkpoint_mutation,
    ensure_workspace_checkpoint,
)
from cayu.workspaces.mutation_attribution import (
    WorkspaceMutationWindow,
    begin_workspace_mutation_window,
)
from cayu.workspaces.observation_recovery import (
    WorkspaceObservationEvidenceState,
    WorkspaceObservationLifecycle,
    WorkspaceObservationPhase,
    WorkspaceObservationTerminalStatus,
    _admit_workspace_observation_intent,
    _project_workspace_observation_authority,
    await_workspace_observation_store_read,
    publish_workspace_observation_transition,
    workspace_observation_event_digest,
    workspace_observations_from_checkpoint,
)
from cayu.workspaces.revisions import (
    WorkspaceRevisionObservation,
    WorkspaceRevisionObservationStatus,
    WorkspaceWriterIsolationEvidence,
)


class ToolInvocation:
    """Execute one call while retaining its authority, evidence and terminal result."""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        event_writer: RuntimeEventWriter,
        auxiliary_inference: AuxiliaryInferenceOwner,
        session_control: SessionControl[SessionUsageTracker],
        admission: ToolInvocationAdmission,
        hooks: ToolInvocationHooks,
        terminals: ToolTerminalPublisher,
        resources: ToolInvocationResources,
        secret_redactor: SecretRedactor,
        tool_timeout_seconds: float | None,
        strict_common_budget_admission: bool = False,
    ) -> None:
        self._session_store = session_store
        self._event_writer = event_writer
        self._auxiliary_inference = auxiliary_inference
        self._session_control = session_control
        self.admission = admission
        self.hooks = hooks
        self.terminals = terminals
        self.resources = resources
        self._secret_redactor = secret_redactor
        self._tool_timeout_seconds = tool_timeout_seconds
        self._strict_common_budget_admission = strict_common_budget_admission
        self._workspace_capture_operations = BoundedInvocationOperationRegistry(
            max_operations=_MAX_RETAINED_WORKSPACE_CAPTURE_OPERATIONS
        )

    @timed_phase("unattributed")
    async def execute(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        tool_call: runtime_records.ToolCallRequest,
        request_metadata: dict[str, Any],
        budget_limits: tuple[BudgetLimit, ...],
        task_id: str | None,
        model_step: int | None = None,
        execution_profile: ExecutionProfileIdentity | None = None,
        invocation_context: InvocationContext | None = None,
        check_policy: bool = True,
        auxiliary_invocation_policy: AuxiliaryInvocationPolicy | None = None,
        emit_started: bool = True,
        policy_result: ToolPolicyResult | None = None,
        policy_evidence: ToolPolicyEvidence = ToolPolicyEvidence.AUTHORITATIVE,
        tool_exposure: ResolvedToolExposureAuthority | None = None,
        policy_output_secret_resolution_scope: Literal["static", "dynamic", "unknown"] = "unknown",
        approval_id: str | None = None,
        tool_round_identity: ToolRoundIdentity,
        input_id: str | None = None,
        taint_labels: frozenset[str] | None = None,
        publish_arguments_as_unavailable: bool = False,
        deferred_terminal_stager: DeferredTerminalStager | None = None,
        deferred_terminal_capture_recorder: DeferredTerminalCaptureRecorder | None = None,
        resolved_redactor_observer: (
            Callable[[str, InvocationRedactorSnapshot], Awaitable[None]] | None
        ) = None,
        publication_snapshot_observer: (
            Callable[
                [str, invocation_secrets.InvocationPublicationSnapshot],
                Awaitable[None],
            ]
            | None
        ) = None,
        rejoin_targeted_invocation: bool = False,
    ) -> AsyncGenerator[tuple[Event, runtime_records.ToolCallOutcome | None], None]:
        if invocation_context is not None and (
            invocation_context.binding.session_id != session.id
            or invocation_context.registered_agent is not registered_agent
            or invocation_context.registered_environment is not registered_environment
            or invocation_context.profile is not execution_profile
        ):
            raise RuntimeError("Tool execution lost frozen invocation authority.")
        tool_round_identity = copy_tool_round_identity(tool_round_identity)
        identity_payload = tool_round_identity.payload()
        tool_round_id = tool_round_identity.tool_round_id
        environment_name = _environment_name(registered_environment)
        if registered_environment is not None:
            if invocation_context is None or execution_profile is None:
                raise RuntimeError(
                    "Environment-backed tool execution requires frozen exposure authority."
                )
            require_environment_exposed(
                registered_environment,
                session=session,
                invocation_context=invocation_context,
                registered_agent=registered_agent,
                execution_profile=execution_profile,
            )
            await ensure_workspace_checkpoint(self._session_store, session, registered_environment)
        registered_tool = registered_agent.executable_tool(tool_call.name)
        if rejoin_targeted_invocation:
            rejoined_events = await self.admission.rejoin_targeted_call(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                tool_call=tool_call,
                task_id=task_id,
                invocation_context=invocation_context,
            )
            for rejoined_event in rejoined_events:
                yield rejoined_event, None
        if tool_call.targeted_tool_rejection is not None:
            rejection = tool_call.targeted_tool_rejection
            result = ToolResult(
                content=targeted_tool_rejection_content(
                    rejection.reason,
                    dispatch_kind=rejection.dispatch_kind,
                ),
                structured={
                    "status": "rejected",
                    "reason": rejection.reason.value,
                },
                is_error=True,
            )
            if (
                rejection.dispatch_kind == "gateway"
                and invocation_context is not None
                and resolve_tool_discovery_projection(
                    registered_agent.tool_discovery_mode,
                    provider=invocation_context.registered_provider.provider,
                    model=invocation_context.binding.model,
                )
                is ToolDiscoveryProjectionKind.SEARCH_TOOLS
            ):
                corrective = dynamic_tool_reference_rejection(rejection.reason)
                result = ToolResult(
                    content=corrective.message,
                    structured=corrective.model_dump(mode="json"),
                    is_error=True,
                )
            idempotency_key = tool_execution.tool_idempotency_key(
                session_id=session.id,
                tool_call_id=tool_call.id,
                tool_round_id=tool_round_id,
                approval_id=approval_id,
                pause_id=input_id,
            )
            payload: dict[str, Any] = {
                "tool_call_id": tool_call.id,
                "idempotency_key": idempotency_key,
                "blocked_by": (
                    "targeted_tool_gateway"
                    if rejection.dispatch_kind == "gateway"
                    else "targeted_tool_native"
                ),
                "reason": rejection.reason.value,
                "rejection_event_id": rejection.rejection_event_id,
                "dispatch_kind": rejection.dispatch_kind,
                "model_tool_name": rejection.model_tool_name,
                "result": result.model_dump(mode="json"),
                **tool_argument_publication.unavailable_argument_projection().payload_fields(),
                **identity_payload,
            }
            event = event_with_runtime_payload_authority(
                _event_with_tool_round_authority(
                    event_with_execution_profile_authority(
                        Event(
                            type=EventType.TOOL_CALL_FAILED,
                            session_id=session.id,
                            agent_name=registered_agent.spec.name,
                            environment_name=environment_name,
                            tool_name=rejection.model_tool_name,
                            payload=payload,
                        ),
                        execution_profile,
                    ),
                    tool_round_identity,
                ),
                "dispatch_kind",
                "model_tool_name",
            )
            outcome = runtime_records.ToolCallOutcome(
                call=replace(tool_call, arguments={}, arguments_state="unavailable"),
                result=result,
            )
            publication_snapshot = invocation_secrets.InvocationPublicationSnapshot(
                redactor=self._secret_redactor,
                unsafe_output=False,
            )
            terminal = await self.terminals.publish_unexecuted(
                tool_call_id=tool_call.id,
                event=event,
                outcome=outcome,
                snapshot=publication_snapshot,
                observer=publication_snapshot_observer,
                stager=deferred_terminal_stager,
            )
            if terminal is not None:
                yield terminal
            return
        mcp_policy_authority_rejected = (
            policy_evidence is ToolPolicyEvidence.UNREGISTERED
            and registered_tool is not None
            and isinstance(registered_tool.tool, McpToolAdapter)
        )
        if mcp_policy_authority_rejected or _registered_mcp_tool_authority_is_unavailable(
            registered_tool
        ):
            result = ToolResult(
                content="MCP tool unavailable because its catalogue authority is not current.",
                structured={
                    "status": "rejected",
                    "reason": "mcp_catalogue_authority_unavailable",
                },
                is_error=True,
            )
            idempotency_key = tool_execution.tool_idempotency_key(
                session_id=session.id,
                tool_call_id=tool_call.id,
                tool_round_id=tool_round_id,
                approval_id=approval_id,
                pause_id=input_id,
            )
            payload: dict[str, Any] = {
                "tool_call_id": tool_call.id,
                "idempotency_key": idempotency_key,
                "blocked_by": "mcp_catalogue_authority",
                "reason": "mcp_catalogue_authority_unavailable",
                "result": result.model_dump(mode="json"),
                **_targeted_tool_invocation_payload(tool_call),
                **tool_argument_publication.unavailable_argument_projection().payload_fields(),
                **identity_payload,
            }
            if approval_id is not None:
                payload["approval_id"] = approval_id
            if input_id is not None:
                payload["input_id"] = input_id
            event = _event_with_targeted_tool_invocation_authority(
                _event_with_tool_round_authority(
                    event_with_execution_profile_authority(
                        Event(
                            type=EventType.TOOL_CALL_FAILED,
                            session_id=session.id,
                            agent_name=registered_agent.spec.name,
                            environment_name=environment_name,
                            tool_name=tool_call.model_tool_name or tool_call.name,
                            payload=payload,
                        ),
                        execution_profile,
                    ),
                    tool_round_identity,
                    *(field for field in ("approval_id", "input_id") if field in payload),
                ),
                tool_call,
            )
            outcome = runtime_records.ToolCallOutcome(
                call=replace(tool_call, arguments={}, arguments_state="unavailable"),
                result=result,
            )
            publication_snapshot = invocation_secrets.InvocationPublicationSnapshot(
                redactor=self._secret_redactor,
                unsafe_output=False,
            )
            terminal = await self.terminals.publish_unexecuted(
                tool_call_id=tool_call.id,
                event=event,
                outcome=outcome,
                snapshot=publication_snapshot,
                observer=publication_snapshot_observer,
                stager=deferred_terminal_stager,
            )
            if terminal is not None:
                yield terminal
            return
        if policy_evidence is ToolPolicyEvidence.UNEXPOSED:
            if tool_exposure is None:
                raise RuntimeError("Unexposed tool call lost its frozen exposure snapshot.")
            tool_exposure = validate_resolved_tool_exposure_authority(
                tool_exposure,
                registered_agent.tool_capabilities,
                catalogue_revision=registered_agent.tool_catalogue.revision,
            )
            if tool_call.name not in registered_agent.executable_tool_names:
                raise RuntimeError("Unexposed tool call is not registered.")
            if tool_call.name in tool_exposure.tool_names:
                raise RuntimeError("Unexposed tool call is present in its exposure snapshot.")
            result = unexposed_tool_result()
            idempotency_key = tool_execution.tool_idempotency_key(
                session_id=session.id,
                tool_call_id=tool_call.id,
                tool_round_id=tool_round_id,
                approval_id=approval_id,
                pause_id=input_id,
            )
            payload: dict[str, Any] = {
                "tool_call_id": tool_call.id,
                "idempotency_key": idempotency_key,
                "blocked_by": "tool_exposure",
                "reason": NOT_EXPOSED_IN_REQUEST_REASON,
                "profile_id": tool_exposure.profile_id,
                "exposure_fingerprint": tool_exposure.fingerprint,
                "result": result.model_dump(mode="json"),
                **tool_argument_publication.unavailable_argument_projection().payload_fields(),
                **identity_payload,
            }
            if approval_id is not None:
                payload["approval_id"] = approval_id
            if input_id is not None:
                payload["input_id"] = input_id
            event = _event_with_tool_round_authority(
                event_with_execution_profile_authority(
                    Event(
                        type=EventType.TOOL_CALL_BLOCKED,
                        session_id=session.id,
                        agent_name=registered_agent.spec.name,
                        environment_name=environment_name,
                        tool_name=tool_call.name,
                        payload=payload,
                    ),
                    execution_profile,
                ),
                tool_round_identity,
                *(
                    field
                    for field in (
                        "approval_id",
                        "input_id",
                        "profile_id",
                        "exposure_fingerprint",
                    )
                    if field in payload
                ),
            )
            projected_call = replace(tool_call, arguments={}, arguments_state="unavailable")
            outcome = runtime_records.ToolCallOutcome(
                call=projected_call,
                result=result,
            )
            publication_snapshot = invocation_secrets.InvocationPublicationSnapshot(
                redactor=self._secret_redactor,
                unsafe_output=False,
            )
            terminal = await self.terminals.publish_unexecuted(
                tool_call_id=tool_call.id,
                event=event,
                outcome=outcome,
                snapshot=publication_snapshot,
                observer=publication_snapshot_observer,
                stager=deferred_terminal_stager,
            )
            if terminal is not None:
                yield terminal
            return
        model_arguments = copy_json_value(tool_call.arguments, "tool_call.arguments")
        invocation_redactor = _redactor_for_tool_calls(
            self._secret_redactor,
            registered_agent=registered_agent,
            tool_calls=[tool_call],
        )
        provisional_output_redactor = invocation_redactor
        publication = InvocationPublication(
            tool_call=tool_call,
            redactor=invocation_redactor,
            observer=publication_snapshot_observer,
        )

        started_event: Event | None = None
        if registered_tool is not None and not registered_tool.publish_arguments:
            publish_arguments_as_unavailable = True
        if taint_labels is None:
            taint_labels = frozenset(
                await self.admission.prior_taint_labels(
                    session_id=session.id,
                    policy=registered_agent.tool_policy,
                    request_metadata=request_metadata,
                )
            )
        idempotency_key = tool_execution.tool_idempotency_key(
            session_id=session.id,
            tool_call_id=tool_call.id,
            tool_round_id=tool_round_id,
            approval_id=approval_id,
            pause_id=input_id,
        )
        argument_error = _registered_tool_argument_error(registered_tool, tool_call.arguments)
        if emit_started and argument_error is None:
            payload: dict[str, Any] = {
                "tool_call_id": tool_call.id,
                "idempotency_key": idempotency_key,
                **_targeted_tool_invocation_payload(tool_call),
                **tool_argument_publication.quarantined_argument_fields(),
                **_published_argument_presence(
                    tool_call, registered_agent.executable_tool(tool_call.name)
                ),
                **identity_payload,
            }
            if registered_tool is not None:
                payload["effect"] = registered_tool.effect.value
            if approval_id is not None:
                payload["approval_id"] = approval_id
            if input_id is not None:
                payload["input_id"] = input_id
            started = event_with_execution_profile_authority(
                Event(
                    type=EventType.TOOL_CALL_STARTED,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    tool_name=tool_call.name,
                    payload=payload,
                ),
                execution_profile,
            )
            started = _event_with_targeted_tool_invocation_authority(started, tool_call)
            started_event = await self._event_writer.emit(
                prepare_runtime_event(
                    _event_with_tool_round_authority(
                        started,
                        tool_round_identity,
                        *(field for field in ("approval_id", "input_id") if field in payload),
                    ),
                    redactor=invocation_redactor,
                )
            )
            yield started_event, None

        if registered_tool is None:
            publication_snapshot = await publication.static_scope()
            result = ToolResult(
                content=f"Tool not registered: {tool_call.name}",
                is_error=True,
            )
            payload = {
                "tool_call_id": tool_call.id,
                "idempotency_key": idempotency_key,
                "result": result.model_dump(),
                **identity_payload,
            }
            if approval_id is not None:
                payload["approval_id"] = approval_id
            if input_id is not None:
                payload["input_id"] = input_id
            async for event in self.terminals.publish_result(
                event=Event(
                    type=EventType.TOOL_CALL_FAILED,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    tool_name=tool_call.name,
                    payload=payload,
                ),
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                tool_call=tool_call,
                result=result,
                task_id=task_id,
                execution_profile=execution_profile,
                invocation_context=invocation_context,
                redactor=invocation_redactor,
                output_redactor=provisional_output_redactor,
                deferred_terminal_stager=deferred_terminal_stager,
                publication_snapshot=publication_snapshot,
            ):
                yield event
            return

        if argument_error is not None:
            # Approval of an older malformed call cannot bypass current input validation.
            check_policy = True
            policy_result = ToolPolicyResult(
                decision=ToolPolicyDecision.DENY,
                reason=argument_error,
                metadata={"reason": "invalid_arguments"},
            )
        if check_policy:
            if policy_result is None:
                resolved_policy_result = await self.admission.authorize(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    tool_call=tool_call,
                    request_metadata=request_metadata,
                    taint_labels=taint_labels,
                )
            else:
                resolved_policy_result = tool_execution.validate_tool_policy_result(policy_result)
            if resolved_policy_result.decision == ToolPolicyDecision.DENY:
                publication_snapshot = await publication.static_scope()
                public_policy_result = (
                    resolved_policy_result
                    if argument_error is not None
                    else approval_support.public_policy_denial_result(
                        secret_resolution_scope=policy_output_secret_resolution_scope,
                        policy_result=resolved_policy_result,
                        publish_arguments=registered_tool.publish_arguments,
                    )
                )
                reason = tool_execution.policy_denial_reason(public_policy_result)
                result = tool_execution.blocked_tool_result(public_policy_result, reason=reason)
                payload = {
                    "tool_call_id": tool_call.id,
                    "idempotency_key": idempotency_key,
                    **policy_denial_payload_fields(
                        tool_name=tool_call.name,
                        denied_by=_TOOL_POLICY_DENIAL_SOURCE,
                        decision=public_policy_result.decision.value,
                        reason=reason,
                        metadata=public_policy_result.metadata,
                    ),
                    "result": result.model_dump(),
                    **identity_payload,
                }
                if approval_id is not None:
                    payload["approval_id"] = approval_id
                if input_id is not None:
                    payload["input_id"] = input_id
                async for event in self.terminals.publish_result(
                    event=Event(
                        type=EventType.TOOL_CALL_BLOCKED,
                        session_id=session.id,
                        agent_name=registered_agent.spec.name,
                        environment_name=environment_name,
                        tool_name=tool_call.name,
                        payload=payload,
                    ),
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    tool_call=tool_call,
                    result=result,
                    task_id=task_id,
                    execution_profile=execution_profile,
                    invocation_context=invocation_context,
                    redactor=invocation_redactor,
                    output_redactor=provisional_output_redactor,
                    deferred_terminal_stager=deferred_terminal_stager,
                    argument_projection=(
                        tool_argument_publication.finalized_argument_projection(
                            tool_call.arguments,
                            redactor=invocation_redactor,
                            scope_finalized=True,
                        )
                        if argument_error is not None
                        and policy_output_secret_resolution_scope == "static"
                        and registered_tool.publish_arguments
                        else None
                    ),
                    publication_snapshot=publication_snapshot,
                ):
                    yield event
                return
            if resolved_policy_result.decision == ToolPolicyDecision.REQUIRE_APPROVAL:
                approval, approval_events = await self.admission.pause_for_approval(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    tool_call=tool_call,
                    tool_calls=[tool_call],
                    policy_outcomes=None,
                    active_taint_by_id={tool_call.id: taint_labels},
                    task_id=task_id,
                    policy_result=resolved_policy_result,
                    structured_output=None,
                    thinking=None,
                    max_steps=None,
                    limits=None,
                    budget_limits=None,
                    retry_policy=None,
                    tool_round_identity=tool_round_identity,
                )
                for approval_event in approval_events:
                    yield approval_event, None
                raise ToolApprovalRequired(approval)
            if resolved_policy_result.decision != ToolPolicyDecision.ALLOW:
                raise ValueError(
                    f"Unsupported tool policy decision: {resolved_policy_result.decision}"
                )

        anchor_event = started_event or Event(
            type=EventType.TOOL_CALL_STARTED,
            session_id=session.id,
            agent_name=registered_agent.spec.name,
            environment_name=environment_name,
            tool_name=tool_call.name,
            payload={"tool_call_id": tool_call.id, **identity_payload},
        )
        before_resolution = _BeforeToolCallResolution(arguments=deepcopy(tool_call.arguments))
        quarantine_pre_execution_publication = (
            policy_output_secret_resolution_scope != "static"
            or not registered_tool.publish_arguments
        )
        async for hook_event in self.hooks.before_call(
            session=session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            tool_call=tool_call,
            anchor_event=anchor_event,
            task_id=task_id,
            execution_profile=execution_profile,
            invocation_context=invocation_context,
            resolution=before_resolution,
            redactor=invocation_redactor,
            output_redactor=provisional_output_redactor,
            quarantine_output=quarantine_pre_execution_publication,
        ):
            yield hook_event, None
        effective_tool_call = (
            tool_call
            if before_resolution.arguments == tool_call.arguments
            else replace(tool_call, arguments=before_resolution.arguments)
        )
        effective_arguments_payload = (
            {"effective_arguments": effective_tool_call.arguments}
            if effective_tool_call is not tool_call
            else {}
        )
        if before_resolution.block_reason is not None:
            publication_snapshot = await publication.static_scope()
            block_reason = (
                before_resolution.block_reason
                if registered_tool.publish_arguments
                else "Tool call blocked by a before_tool_call hook."
            )
            async for event in self.terminals.emit_result(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                tool_call=effective_tool_call,
                event_type=EventType.TOOL_CALL_BLOCKED,
                result=ToolResult(content=block_reason, is_error=True),
                extra_payload={
                    "reason": block_reason,
                    "blocked_by": "before_tool_call_hook",
                    "idempotency_key": idempotency_key,
                    **effective_arguments_payload,
                },
                task_id=task_id,
                execution_profile=execution_profile,
                invocation_context=invocation_context,
                tool_round_identity=tool_round_identity,
                approval_id=approval_id,
                input_id=input_id,
                allow_modification=False,
                redactor=invocation_redactor,
                output_redactor=provisional_output_redactor,
                deferred_terminal_stager=deferred_terminal_stager,
                publication_snapshot=publication_snapshot,
            ):
                yield event
            return
        if before_resolution.short_circuit_result is not None:
            publication_snapshot = await publication.static_scope()
            short_result = (
                before_resolution.short_circuit_result
                if registered_tool.publish_arguments
                else _private_argument_short_circuit_result(before_resolution.short_circuit_result)
            )
            async for event in self.terminals.emit_result(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                tool_call=effective_tool_call,
                event_type=(
                    EventType.TOOL_CALL_FAILED
                    if short_result.is_error
                    else EventType.TOOL_CALL_COMPLETED
                ),
                result=short_result,
                extra_payload={
                    "short_circuited_by": "before_tool_call_hook",
                    "idempotency_key": idempotency_key,
                    **effective_arguments_payload,
                },
                task_id=task_id,
                execution_profile=execution_profile,
                invocation_context=invocation_context,
                tool_round_identity=tool_round_identity,
                approval_id=approval_id,
                input_id=input_id,
                allow_modification=registered_tool.publish_arguments,
                redactor=invocation_redactor,
                output_redactor=provisional_output_redactor,
                deferred_terminal_stager=deferred_terminal_stager,
                publication_snapshot=publication_snapshot,
            ):
                yield event
            return

        if effective_tool_call is not tool_call:
            reauthorization = await self.admission.authorize(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                tool_call=effective_tool_call,
                request_metadata={
                    **request_metadata,
                    TOOL_POLICY_REAUTHORIZATION_METADATA_KEY: True,
                },
                taint_labels=taint_labels,
            )
            if reauthorization.decision != ToolPolicyDecision.ALLOW:
                publication_snapshot = await publication.static_scope()
                if reauthorization.decision == ToolPolicyDecision.DENY:
                    public_reauthorization = (
                        reauthorization
                        if _registered_tool_argument_error(
                            registered_tool, effective_tool_call.arguments
                        )
                        is not None
                        else approval_support.public_policy_denial_result(
                            secret_resolution_scope=policy_output_secret_resolution_scope,
                            policy_result=reauthorization,
                            publish_arguments=registered_tool.publish_arguments,
                        )
                    )
                    reason = tool_execution.policy_denial_reason(public_reauthorization)
                    metadata = public_reauthorization.metadata
                else:
                    metadata = (
                        reauthorization.metadata
                        if policy_output_secret_resolution_scope == "static"
                        and registered_tool.publish_arguments
                        else {}
                    )
                    reason = (
                        (
                            reauthorization.reason
                            if policy_output_secret_resolution_scope == "static"
                            and registered_tool.publish_arguments
                            else None
                        )
                        or "Modified tool arguments require approval, which before_tool_call "
                        "hook modifications do not support."
                    )
                async for event in self.terminals.emit_result(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    tool_call=effective_tool_call,
                    event_type=EventType.TOOL_CALL_BLOCKED,
                    result=ToolResult(content=reason, is_error=True),
                    extra_payload={
                        **policy_denial_payload_fields(
                            tool_name=effective_tool_call.name,
                            denied_by=_TOOL_POLICY_DENIAL_SOURCE,
                            decision=reauthorization.decision.value,
                            reason=reason,
                            metadata=metadata,
                        ),
                        "blocked_by": "tool_policy_reauthorization",
                        "idempotency_key": idempotency_key,
                    },
                    task_id=task_id,
                    execution_profile=execution_profile,
                    invocation_context=invocation_context,
                    tool_round_identity=tool_round_identity,
                    approval_id=approval_id,
                    input_id=input_id,
                    allow_modification=False,
                    redactor=invocation_redactor,
                    output_redactor=provisional_output_redactor,
                    deferred_terminal_stager=deferred_terminal_stager,
                    publication_snapshot=publication_snapshot,
                ):
                    yield event
                return

        invocation_secret_scope = invocation_secrets.InvocationSecretTracker(invocation_redactor)
        proxy_authorizations: list[invocation_secrets.ProxyAuthorizationRecord] = []
        ctx_metadata = tool_execution.context_metadata(
            request_metadata=request_metadata,
            tool_call_id=tool_call.id,
            approval_id=approval_id,
            idempotency_key=idempotency_key,
            tool_effect=registered_tool.effect,
            input_id=input_id,
        )
        if taint_labels:
            ctx_metadata[TAINT_LABELS_METADATA_KEY] = sorted(taint_labels)
        if execution_profile is not None:
            ctx_metadata[EXECUTION_PROFILE_FINGERPRINT_FIELD] = execution_profile.fingerprint

        invocation_call = ToolInvocationCall(
            session=session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            registered_tool=registered_tool,
            tool_call=tool_call,
            effective_tool_call=effective_tool_call,
            tool_round_identity=tool_round_identity,
            execution_profile=execution_profile,
            invocation_context=invocation_context,
            task_id=task_id,
            budget_limits=budget_limits,
            tool_exposure=tool_exposure,
            environment_name=environment_name,
            idempotency_key=idempotency_key,
            model_step=model_step,
            approval_id=approval_id,
            input_id=input_id,
        )
        evidence = InvocationEvidence(
            call=invocation_call,
            secret_scope=invocation_secret_scope,
            publication=publication,
            session_store=self._session_store,
            event_writer=self._event_writer,
            resolved_redactor_observer=resolved_redactor_observer,
        )
        runner_events = evidence.runner_events
        invocation_resources = self.resources.bind(
            invocation_call,
            ctx_metadata=ctx_metadata,
            invocation_secret_scope=invocation_secret_scope,
            proxy_authorizations=proxy_authorizations,
            redactor_provider=evidence.redactor,
            observe_runner_execution=evidence.observe_runner,
            persist_resolved_secret_projection=evidence.persist_resolved,
        )
        tool_context = invocation_resources.context
        raw_workspace = invocation_resources.raw_workspace
        direct_workspace_mutations = invocation_resources.direct_workspace_mutations
        workspace_mutation_owner = invocation_resources.workspace_mutation_owner
        workspace_receipt_artifact_store = invocation_resources.workspace_receipt_artifact_store
        workspace_receipt_artifact_unavailable_detail = (
            invocation_resources.workspace_receipt_artifact_unavailable_detail
        )
        workspace_window_id: str | None = None
        workspace_attribution_window: WorkspaceMutationWindow | None = None
        writer_isolation_before = WorkspaceWriterIsolationEvidence()
        before_workspace_observation: WorkspaceRevisionObservation | None = None
        workspace_lifecycle: WorkspaceObservationLifecycle | None = None
        if registered_tool.workspace_mutation and _workspace(registered_environment) is not None:
            if registered_tool.parallel_safe:
                raise RuntimeError("Workspace-mutating tools must execute exclusively.")
            if deferred_terminal_stager is None or deferred_terminal_capture_recorder is None:
                raise RuntimeError("Workspace-mutating tools require durable terminal staging.")
            if registered_environment is None:  # pragma: no cover - narrowed by workspace lookup
                raise AssertionError("Workspace mutation lost its registered environment.")
            workspace_window_id = _workspace_mutation_window_id(
                session_id=session.id,
                session_run_epoch=session.run_epoch,
                tool_round_id=tool_round_identity.tool_round_id,
                tool_call_id=tool_call.id,
            )
            await begin_workspace_checkpoint_mutation(
                self._session_store,
                session,
                registered_environment,
                window_id=workspace_window_id,
                tool_call_id=tool_call.id,
                interaction_id=None if started_event is None else started_event.interaction_id,
            )
            configured_workspace_id = _workspace_id(registered_environment)
            configured_observer = _workspace_revision_observer_name(registered_environment)
            configured_artifact_store_id = _artifact_store_id(registered_environment)
            workspace_authority = _project_workspace_observation_authority(
                session_id=session.id,
                configured_workspace_id=configured_workspace_id,
                configured_observer=configured_observer,
                configured_artifact_store_id=configured_artifact_store_id,
                observer_is_runtime_owned=(
                    _workspace_revision_observer_is_runtime_owned(registered_environment)
                ),
                secret_resolution_scope=(
                    invocation_secrets.registered_environment_secret_resolution_scope(
                        registered_environment
                    )
                ),
                redactor=self._secret_redactor,
                public_authority_alias_codec=(self._session_store.public_authority_alias_codec),
            )
            workspace_lifecycle = WorkspaceObservationLifecycle(
                session_id=session.id,
                interaction_id=(None if started_event is None else started_event.interaction_id),
                window_id=workspace_window_id,
                source_run_epoch=session.run_epoch,
                binding_generation_id=_workspace_binding_generation_id(registered_environment),
                workspace_id=workspace_authority.workspace_id,
                observer=workspace_authority.observer,
                observer_authority=workspace_authority.observer_authority,
                artifact_store_id=workspace_authority.artifact_store_id,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                model_step_id=tool_round_identity.model_step_id,
                model_attempt_id=tool_round_identity.model_attempt_id,
                tool_round_id=tool_round_identity.tool_round_id,
                model_step=model_step,
            )
            await publish_workspace_observation_transition(
                session_store=self._session_store,
                event_writer=self._event_writer,
                session=session,
                previous=None,
                current=workspace_lifecycle,
                phase="intent",
                intent_admission=_admit_workspace_observation_intent(
                    workspace_lifecycle,
                    redactor=self._secret_redactor,
                    configured_workspace_id=configured_workspace_id,
                    configured_artifact_store_id=configured_artifact_store_id,
                    authority_projection=workspace_authority,
                ),
            )
            workspace_attribution_window = begin_workspace_mutation_window(
                raw_workspace,
                window_id=workspace_window_id,
            )
            writer_isolation_before = _workspace_writer_isolation(registered_environment)
            try:
                before_workspace_observation = await _observe_workspace_revision(
                    registered_environment,
                    operation_registry=self._workspace_capture_operations,
                    require_quiescence_before_return=True,
                    authority_projection=workspace_authority,
                )
                captured_before_state = (
                    WorkspaceObservationEvidenceState.FAILED
                    if before_workspace_observation.status
                    is WorkspaceRevisionObservationStatus.FAILED
                    else WorkspaceObservationEvidenceState.CAPTURED_PRIVATE
                )
                before_captured_lifecycle = WorkspaceObservationLifecycle.model_validate(
                    {
                        **workspace_lifecycle.model_dump(mode="json"),
                        "phase": WorkspaceObservationPhase.BEFORE_CAPTURED.value,
                        "before_state": captured_before_state.value,
                    }
                )
                await publish_workspace_observation_transition(
                    session_store=self._session_store,
                    event_writer=self._event_writer,
                    session=session,
                    previous=workspace_lifecycle,
                    current=before_captured_lifecycle,
                    phase="before-capture",
                )
            except BaseException:
                workspace_attribution_window.close()
                raise
            workspace_lifecycle = before_captured_lifecycle

        async def close_workspace_mutation_window() -> tuple[Event, ...]:
            if workspace_window_id is None or before_workspace_observation is None:
                return ()
            if workspace_mutation_owner is not None:
                await workspace_mutation_owner.seal_and_wait()
            return ()

        def retain_workspace_settlement_failure(interrupt: BaseException) -> None:
            safe_failure = RuntimeError(
                "Workspace mutation settlement failed after caller cancellation."
            )
            prior_cause = exception_cause(interrupt)
            cause: BaseException = safe_failure
            if prior_cause is not None:
                cause = BaseExceptionGroup(
                    "Caller cancellation and workspace mutation settlement failures.",
                    [prior_cause, safe_failure],
                )
            if isinstance(interrupt, asyncio.CancelledError):
                attach_runner_cancellation_failure(interrupt, cause)
            elif isinstance(interrupt, BaseExceptionGroup):
                for candidate in iter_exception_tree(interrupt):
                    if isinstance(candidate, asyncio.CancelledError):
                        attach_runner_cancellation_failure(candidate, cause)
                        set_exception_cause(candidate, cause)
            set_exception_cause(interrupt, cause)

        async def close_workspace_mutation_window_after_interrupt(
            interrupt: BaseException,
            cancellation: asyncio.CancelledError,
        ) -> None:

            nonlocal workspace_capture_failure_detail, workspace_settlement_failure
            outcome = await await_invocation_operation(
                close_workspace_mutation_window,
                request_child_cancellation=False,
                cancellation=cancellation,
                on_unsettled_supervisory_exit=(
                    workspace_mutation_owner.fail_closed
                    if workspace_mutation_owner is not None
                    else None
                ),
            )
            if _contains_process_signal(outcome.error):
                if outcome.error is None:  # pragma: no cover - narrowed by the helper
                    raise AssertionError("Process-signal classification lost its error.")
                raise outcome.error
            if outcome.error is not None:
                if any(
                    isinstance(candidate, WorkspaceMutationSettlementError)
                    for candidate in iter_exception_tree(outcome.error)
                ):
                    workspace_settlement_failure = WorkspaceMutationSettlementError(
                        "Workspace mutation settlement could not be proven."
                    )
                else:
                    workspace_capture_failure_detail = "receipt_publication_failed"
                retain_workspace_settlement_failure(interrupt)

        workspace_events: tuple[Event, ...] = ()
        post_tool_cancellation: asyncio.CancelledError | None = None
        post_tool_cancellation_requests_consumed = 0
        workspace_settlement_failure: WorkspaceMutationSettlementError | None = None
        workspace_capture_failure_detail: str | None = None
        workspace_capture_payload: dict[str, str] = {}

        async def consume_post_tool_cancellation(
            cancellation: asyncio.CancelledError,
        ) -> asyncio.CancelledError | None:

            nonlocal post_tool_cancellation_requests_consumed
            current_task = asyncio.current_task()
            await _receive_restored_post_tool_cancellation()
            requests_before = 0 if current_task is None else current_task.cancelling()
            observed = consume_pending_task_cancellation(cancellation)
            if current_task is not None:
                post_tool_cancellation_requests_consumed += max(
                    requests_before - current_task.cancelling(),
                    0,
                )
            return observed

        effective_terminal_stager = deferred_terminal_stager
        if workspace_lifecycle is not None:
            if (
                deferred_terminal_stager is None
                or deferred_terminal_capture_recorder is None
                or workspace_window_id is None
                or workspace_attribution_window is None
                or before_workspace_observation is None
            ):
                raise RuntimeError("Workspace observation lost its durable staging boundary.")

            async def stage_workspace_terminal(
                event: Event,
                outcome: runtime_records.ToolCallOutcome,
                allow_modification: bool,
                publish_before_hooks: bool,
                snapshot: invocation_secrets.InvocationPublicationSnapshot,
            ) -> Event:

                nonlocal post_tool_cancellation
                nonlocal workspace_events, workspace_lifecycle, workspace_capture_payload
                uncertainty_error: ToolEffectReconciliationRequired | None = None
                try:
                    if registered_environment is not None and workspace_window_id is not None:
                        await complete_workspace_checkpoint_mutation(
                            self._session_store,
                            session,
                            registered_environment,
                            window_id=workspace_window_id,
                            window_exclusive=lambda: (
                                not workspace_attribution_window.overlap_detected
                            ),
                            successful=(
                                event.type is EventType.TOOL_CALL_COMPLETED
                                and not outcome.result.is_error
                                and not snapshot.unsafe_output
                                and workspace_settlement_failure is None
                            ),
                        )
                    try:
                        staged_event = await deferred_terminal_stager(
                            event,
                            outcome,
                            allow_modification,
                            publish_before_hooks,
                            snapshot,
                        )
                    except ToolEffectReconciliationRequired as exc:
                        uncertainty_error = exc
                        effect_owner = ToolEffectStateOwner(self._session_store)
                        effect = await effect_owner.resolve_call(
                            session,
                            tool_round_id=tool_round_identity.tool_round_id,
                            tool_call_id=tool_call.id,
                        )
                        if effect is not None and effect.state == "prepared":
                            # No dispatch was consumed, hence no uncertainty event
                            # exists to bind to this observation. Keep the durable
                            # lifecycle for recovery, which owns prepared settlement.
                            workspace_attribution_window.close(discard_history=True)
                            raise
                        uncertain_events = await effect_owner.load_uncertainty_events(
                            session,
                            tool_round_id=tool_round_identity.tool_round_id,
                            tool_call_ids=(tool_call.id,),
                        )
                        if len(uncertain_events) != 1:
                            raise RuntimeError(
                                "Workspace capture lost exact uncertainty evidence."
                            ) from exc
                        staged_event = uncertain_events[0]
                    current_workspace_lifecycle = workspace_lifecycle
                    if current_workspace_lifecycle is None:
                        raise RuntimeError(
                            "Workspace observation lifecycle disappeared before staging."
                        )
                    staged_lifecycle = WorkspaceObservationLifecycle.model_validate(
                        {
                            **current_workspace_lifecycle.model_dump(mode="json"),
                            "phase": WorkspaceObservationPhase.TOOL_OUTCOME_STAGED.value,
                            "tool_outcome_event_id": staged_event.id,
                            "tool_outcome_event_digest": workspace_observation_event_digest(
                                staged_event
                            ),
                        }
                    )
                    await publish_workspace_observation_transition(
                        session_store=self._session_store,
                        event_writer=self._event_writer,
                        session=session,
                        previous=current_workspace_lifecycle,
                        current=staged_lifecycle,
                        phase="tool-outcome",
                    )
                    workspace_lifecycle = staged_lifecycle

                    capture: _WorkspaceCaptureResult | None = None
                    capture_failure_detail: str | None = None
                    workspace_evidence_available = (
                        not snapshot.unsafe_output and not publish_arguments_as_unavailable
                    )
                    if workspace_settlement_failure is not None:
                        capture_failure_detail = "mutation_settlement_unproven"
                    elif workspace_capture_failure_detail is not None:
                        capture_failure_detail = workspace_capture_failure_detail
                    else:
                        try:
                            capture = await _record_workspace_mutation_after(
                                session_store=self._session_store,
                                event_writer=self._event_writer,
                                registered_environment=registered_environment,
                                artifact_store=workspace_receipt_artifact_store,
                                artifact_unavailable_detail_code=(
                                    workspace_receipt_artifact_unavailable_detail
                                ),
                                session=session,
                                registered_agent=registered_agent,
                                environment_name=environment_name,
                                tool_call=tool_call,
                                tool_round_identity=tool_round_identity,
                                execution_profile=execution_profile,
                                model_step=model_step,
                                window_id=workspace_window_id,
                                lifecycle=workspace_lifecycle,
                                before_observation=before_workspace_observation,
                                authority_projection=workspace_authority,
                                attribution_window=workspace_attribution_window,
                                writer_isolation_before=writer_isolation_before,
                                direct_mutations=direct_workspace_mutations,
                                redactor=snapshot.redactor,
                                evidence_available=workspace_evidence_available,
                                operation_registry=self._workspace_capture_operations,
                            )
                        except (KeyboardInterrupt, SystemExit, GeneratorExit):
                            raise
                        except asyncio.CancelledError as exc:
                            observed_cancellation = await consume_post_tool_cancellation(exc)
                            if observed_cancellation is None:
                                raise
                            publication_failure = exception_cause(observed_cancellation)
                            if publication_failure is not None:
                                # The parallel tool-round boundary deliberately
                                # severs ordinary exception causes before it
                                # rebuilds the caller cancellation. Carry this
                                # runtime-owned publication failure through its
                                # authenticated cleanup channel so both the
                                # initial and reconciliation failures survive in
                                # bounded form on the final cancellation.
                                prior_failure = runner_cancellation_failure(observed_cancellation)
                                cancellation_failure = publication_failure
                                if (
                                    prior_failure is not None
                                    and prior_failure is not publication_failure
                                ):
                                    cancellation_failure = BaseExceptionGroup(
                                        "Tool cancellation and workspace publication failures.",
                                        [prior_failure, publication_failure],
                                    )
                                attach_runner_cancellation_failure(
                                    observed_cancellation,
                                    cancellation_failure,
                                )
                            retain_cancellation_context(
                                observed_cancellation,
                                redactor=snapshot.redactor,
                                tool_call_id=tool_call.id,
                            )
                            post_tool_cancellation = observed_cancellation
                            capture_failure_detail = "receipt_publication_interrupted"
                        except Exception:
                            capture_failure_detail = "receipt_publication_failed"
                        finally:
                            if workspace_attribution_window is not None:
                                workspace_attribution_window.close(
                                    discard_history=not workspace_evidence_available
                                )

                    if workspace_attribution_window is not None:
                        workspace_attribution_window.close(
                            discard_history=not workspace_evidence_available
                        )

                    if capture is None:
                        if capture_failure_detail == "receipt_publication_interrupted":
                            workspace_capture_payload = {
                                "workspace_mutation_capture_status": "interrupted",
                                "workspace_mutation_capture_detail_code": capture_failure_detail,
                            }
                        else:
                            workspace_capture_payload = {
                                "workspace_mutation_capture_status": "failed",
                                "workspace_mutation_capture_detail_code": (
                                    capture_failure_detail or "receipt_publication_failed"
                                ),
                            }
                    else:
                        workspace_capture_payload = {
                            "workspace_mutation_capture_status": "recorded",
                        }

                    if capture is None:
                        final_payload = dict(staged_event.payload)
                        final_payload.update(workspace_capture_payload)
                        finalized_stage = (
                            staged_event
                            if uncertainty_error is not None
                            else await deferred_terminal_capture_recorder(
                                staged_event.model_copy(
                                    update={"payload": final_payload}, deep=True
                                )
                            )
                        )
                        if (
                            finalized_stage.id != staged_event.id
                            or workspace_observation_event_digest(finalized_stage)
                            != staged_lifecycle.tool_outcome_event_digest
                        ):
                            raise RuntimeError(
                                "Workspace capture changed the authoritative tool outcome."
                            )
                        checkpoint = await await_workspace_observation_store_read(
                            lambda: self._session_store.load_checkpoint(session.id),
                            operation="Workspace observation failure-closure checkpoint read",
                        )
                        current_lifecycle = workspace_observations_from_checkpoint(checkpoint).get(
                            workspace_window_id
                        )
                        if current_lifecycle is None:
                            raise RuntimeError(
                                "Workspace observation lifecycle disappeared before failure closure."
                            )
                        terminal_lifecycle = _workspace_observation_terminal_view(current_lifecycle)
                        incomplete_event = prepare_runtime_event(
                            _workspace_mutation_incomplete_event(
                                lifecycle=terminal_lifecycle,
                                session=session,
                                execution_profile=execution_profile,
                                status=(
                                    WorkspaceObservationTerminalStatus.INCOMPLETE
                                    if capture_failure_detail == "receipt_publication_interrupted"
                                    else WorkspaceObservationTerminalStatus.FAILED
                                ),
                                detail_code=(
                                    capture_failure_detail or "receipt_publication_failed"
                                ),
                            ),
                            redactor=snapshot.redactor,
                        )
                        published = await publish_workspace_observation_transition(
                            session_store=self._session_store,
                            event_writer=self._event_writer,
                            session=session,
                            previous=current_lifecycle,
                            current=None,
                            phase="terminal",
                            terminal_status=(
                                WorkspaceObservationTerminalStatus.INCOMPLETE
                                if capture_failure_detail == "receipt_publication_interrupted"
                                else WorkspaceObservationTerminalStatus.FAILED
                            ),
                            terminal_detail_code=(
                                capture_failure_detail or "receipt_publication_failed"
                            ),
                            terminal_artifacts=terminal_lifecycle.artifacts,
                            events=(incomplete_event,),
                        )
                        workspace_events = published
                        workspace_lifecycle = current_lifecycle
                        if uncertainty_error is not None:
                            raise uncertainty_error
                        return finalized_stage

                    workspace_lifecycle = capture.lifecycle
                    delta_state = (
                        WorkspaceObservationEvidenceState.PUBLISHED
                        if capture.terminal_status is not WorkspaceObservationTerminalStatus.FAILED
                        else WorkspaceObservationEvidenceState.FAILED
                    )
                    delta_published = WorkspaceObservationLifecycle.model_validate(
                        {
                            **capture.delta_lifecycle.model_dump(mode="json"),
                            "phase": WorkspaceObservationPhase.DELTA_PUBLISHED.value,
                            "delta_state": delta_state.value,
                            "mutation_event_id": capture.receipt_event.id,
                            "mutation_event_digest": workspace_observation_event_digest(
                                capture.receipt_event
                            ),
                        }
                    )
                    (receipt_event,) = await publish_workspace_observation_transition(
                        session_store=self._session_store,
                        event_writer=self._event_writer,
                        session=session,
                        previous=capture.lifecycle,
                        current=delta_published,
                        phase="delta-publication",
                        events=(capture.receipt_event,),
                    )
                    final_payload = dict(staged_event.payload)
                    final_payload.update(workspace_capture_payload)
                    finalized_stage = (
                        staged_event
                        if uncertainty_error is not None
                        else await deferred_terminal_capture_recorder(
                            staged_event.model_copy(update={"payload": final_payload}, deep=True)
                        )
                    )
                    if (
                        finalized_stage.id != staged_event.id
                        or workspace_observation_event_digest(finalized_stage)
                        != staged_lifecycle.tool_outcome_event_digest
                    ):
                        raise RuntimeError(
                            "Workspace capture changed the authoritative tool outcome."
                        )
                    terminal_lifecycle = _workspace_observation_terminal_view(delta_published)
                    finalized_event = prepare_runtime_event(
                        _workspace_mutation_incomplete_event(
                            lifecycle=terminal_lifecycle,
                            session=session,
                            execution_profile=execution_profile,
                            status=capture.terminal_status,
                            detail_code=capture.terminal_detail_code,
                        ),
                        redactor=snapshot.redactor,
                    )
                    (finalized_event,) = await publish_workspace_observation_transition(
                        session_store=self._session_store,
                        event_writer=self._event_writer,
                        session=session,
                        previous=delta_published,
                        current=None,
                        phase="terminal",
                        terminal_status=capture.terminal_status,
                        terminal_detail_code=capture.terminal_detail_code,
                        terminal_artifacts=terminal_lifecycle.artifacts,
                        events=(finalized_event,),
                    )
                    workspace_lifecycle = delta_published
                    workspace_events = (*capture.events, receipt_event, finalized_event)
                    if uncertainty_error is not None:
                        raise uncertainty_error
                    return finalized_stage
                except Exception as cleanup_error:
                    if uncertainty_error is None or cleanup_error is uncertainty_error:
                        raise
                    raise ToolEffectReconciliationCleanupFailure(
                        uncertainty_error,
                        cast("Exception", sanitize_runner_failure(cleanup_error)),
                    ) from None

            effective_terminal_stager = stage_workspace_terminal

        async def stage_interrupted_workspace_outcome(
            interrupt: BaseException,
            *,
            artifacts: list[dict[str, Any]] | None,
        ) -> None:
            """Stage and close exact workspace evidence before re-raising interruption."""

            if workspace_lifecycle is None:
                return
            if effective_terminal_stager is None:
                raise RuntimeError("Workspace interruption lost its terminal staging boundary.")
            interrupted_outcome = _interrupted_tool_call_outcome(
                tool_call=tool_call,
                tool_round_identity=tool_round_identity,
                registered_tool=registered_agent.executable_tool(tool_call.name),
                execution_started=True,
                artifacts=artifacts,
            )
            interrupted_event = _interrupted_tool_call_event(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                tool_call_outcome=interrupted_outcome,
                tool_round_identity=tool_round_identity,
            )
            try:
                snapshot = invocation_secret_scope.seal_for_publication()
                projection = (
                    tool_argument_publication.unavailable_argument_projection()
                    if publish_arguments_as_unavailable
                    else tool_argument_publication.finalized_argument_projection(
                        model_arguments,
                        redactor=snapshot.redactor,
                        scope_finalized=not snapshot.unsafe_output,
                    )
                )
                interrupted_payload = {
                    **interrupted_event.payload,
                    **projection.payload_fields(),
                    tool_argument_publication.ARGUMENTS_EXACT_FIELD: (
                        tool_argument_publication.argument_projection_is_exact(
                            projection,
                            private_arguments=model_arguments,
                        )
                    ),
                }
                if projection.state == "finalized" and effective_arguments_payload:
                    interrupted_payload["effective_arguments"] = snapshot.redactor.redact_json(
                        effective_tool_call.arguments
                    )
                interrupted_event = interrupted_event.model_copy(
                    update={"payload": interrupted_payload}
                )
                await effective_terminal_stager(
                    interrupted_event,
                    interrupted_outcome,
                    False,
                    False,
                    snapshot,
                )
            except BaseException as closure_error:
                if isinstance(closure_error, ToolEffectReconciliationCleanupFailure):
                    closure_error = closure_error.cleanup
                elif isinstance(closure_error, ToolEffectReconciliationRequired):
                    # Closure completed or prepared observation awaits recovery;
                    # neither case adds a failure to the original cancellation.
                    return
                if _contains_process_signal(closure_error):
                    interrupt_nodes = {
                        id(candidate) for candidate in iter_exception_tree(interrupt)
                    }
                    closure_nodes = {
                        id(candidate) for candidate in iter_exception_tree(closure_error)
                    }
                    if interrupt_nodes & closure_nodes:
                        raise
                    raise BaseExceptionGroup(
                        "Tool interruption and workspace observation process control.",
                        [interrupt, closure_error],
                    ) from None
                # The original interruption remains authoritative. Recovery
                # can reconcile an intent or partial phase without rerunning
                # the tool. Retain bounded typed failure evidence, never raw
                # extension messages, tracebacks, or mutable exception state.
                failure: BaseException = sanitize_runner_failure(closure_error)
                prior = exception_cause(interrupt)
                if prior is not None:
                    failure = BaseExceptionGroup(
                        "Tool interruption and workspace observation closure failures.",
                        [prior, failure],
                    )
                # TaskGroup reconstruction clears unauthenticated causal
                # chains. Carry this fixed, runtime-owned diagnostic through
                # the same channel used by mutation settlement cleanup.
                for candidate in iter_exception_tree(interrupt):
                    if isinstance(candidate, asyncio.CancelledError):
                        attach_runner_cancellation_failure(candidate, failure)
                set_exception_cause(interrupt, failure)

        dispatch = ToolInvocationDispatch(
            call=invocation_call,
            context=tool_context,
            secret_scope=invocation_secret_scope,
            session_store=self._session_store,
            auxiliary_inference=self._auxiliary_inference,
            auxiliary_policy=auxiliary_invocation_policy,
            tool_timeout_seconds=self._tool_timeout_seconds,
            strict_common_budget_admission=self._strict_common_budget_admission,
            runner_events=runner_events,
            settle_workspace=(
                close_workspace_mutation_window if workspace_window_id is not None else None
            ),
        )
        try:
            execution_outcome = await dispatch.run()
            for auxiliary_event in dispatch.events:
                yield auxiliary_event, None
        except tool_execution.ToolDispatchAdmissionRefusal as refused:
            refusal = refused.refusal
            if dispatch.effect is not None:
                try:
                    # This proof comes from the runtime callback, never a
                    # tool-authored result or exception. Selection and its
                    # evidence share the existing effect transaction.
                    await self.terminals.settle_admission_refusal(
                        effect=dispatch.effect,
                        session_store=self._session_store,
                        run_epoch=session.run_epoch,
                        execution_profile=execution_profile,
                        tool_round_identity=tool_round_identity,
                    )
                except asyncio.CancelledError as cancellation:
                    secondary = exception_cause(cancellation)
                    invocation_secrets.retain_admission_refusal(
                        cancellation, refusal, settlement_failure=secondary
                    )
                    # Approval and user-input continuations do not cross the
                    # ordinary round's final sanitizer. Publish the same safe
                    # diagnostic snapshot at this shared refusal boundary.
                    invocation_secrets.sanitize_external_cancellation(cancellation)
                    raise cancellation from exception_cause(cancellation)
                except BaseException as settlement_failure:
                    raise BaseExceptionGroup(
                        "Execution admission refusal and settlement failure.",
                        [refusal, settlement_failure],
                    ) from None
            raise refusal from None
        except BaseExceptionGroup as exc:
            if any(
                isinstance(candidate, (KeyboardInterrupt, SystemExit, GeneratorExit))
                for candidate in iter_exception_tree(exc)
            ):
                if workspace_mutation_owner is not None:
                    workspace_mutation_owner.seal_and_transfer_pending()
            elif is_current_runner_cancellation_group(exc):
                group_cancellation: asyncio.CancelledError | None = None
                for candidate in iter_exception_tree(exc):
                    if not isinstance(candidate, asyncio.CancelledError):
                        continue
                    if group_cancellation is None:
                        group_cancellation = candidate
                    retain_cancellation_context(
                        candidate,
                        redactor=invocation_secret_scope.redactor,
                        tool_call_id=tool_call.id,
                    )
                if group_cancellation is None:  # pragma: no cover - classification invariant
                    raise AssertionError(
                        "Current cancellation group has no cancellation leaf."
                    ) from None
                await close_workspace_mutation_window_after_interrupt(
                    exc,
                    group_cancellation,
                )
                await evidence.persist_interrupted(exc)
                (
                    grouped_artifacts,
                    grouped_artifacts_by_id,
                    _grouped_redactors,
                ) = _grouped_cancellation_evidence(exc, tool_calls=[tool_call])
                await stage_interrupted_workspace_outcome(
                    exc,
                    artifacts=(
                        grouped_artifacts_by_id.get(tool_call.id, [])
                        if grouped_artifacts_by_id is not None
                        else grouped_artifacts
                    ),
                )
            raise
        except asyncio.CancelledError as exc:
            retain_cancellation_context(
                exc,
                redactor=invocation_secret_scope.redactor,
                tool_call_id=tool_call.id,
            )
            await close_workspace_mutation_window_after_interrupt(exc, exc)
            cancellation_redelivered = await evidence.persist_interrupted(exc)
            await stage_interrupted_workspace_outcome(
                exc,
                artifacts=invocation_secrets.cancellation_artifacts(exc),
            )
            if cancellation_redelivered:
                raise
            if proxy_authorizations and await self._session_control.interrupt_requested(session.id):
                clear_current_task_cancellation()
                async for event in self._emit_proxy_authorization_events(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    tool_call=tool_call,
                    records=proxy_authorizations,
                    tool_round_identity=tool_round_identity,
                    execution_profile=execution_profile,
                    approval_id=approval_id,
                    input_id=input_id,
                    idempotency_key=idempotency_key,
                    redactor=invocation_secret_scope.redactor,
                    output_redactor=invocation_secret_scope.redactor,
                    quarantine_output=publish_arguments_as_unavailable,
                ):
                    yield event, None
            raise
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            if workspace_mutation_owner is not None:
                workspace_mutation_owner.seal_and_transfer_pending()
            raise

        result_event: Event | None = None
        try:
            result = execution_outcome.result
            redactor = invocation_secret_scope.redactor
            output_redactor = redactor
            if workspace_window_id is not None:
                try:
                    await close_workspace_mutation_window()
                except BaseExceptionGroup as exc:
                    if _contains_process_signal(exc) or is_current_runner_cancellation_group(exc):
                        raise
                    workspace_capture_failure_detail = "receipt_publication_failed"
                    workspace_capture_payload = {
                        "workspace_mutation_capture_status": "failed",
                        "workspace_mutation_capture_detail_code": workspace_capture_failure_detail,
                    }
                except asyncio.CancelledError as exc:
                    post_tool_cancellation = await consume_post_tool_cancellation(exc)
                    if post_tool_cancellation is not None:
                        retain_cancellation_context(
                            post_tool_cancellation,
                            redactor=invocation_secret_scope.redactor,
                            tool_call_id=tool_call.id,
                        )
                    workspace_capture_payload = {
                        "workspace_mutation_capture_status": "interrupted",
                        "workspace_mutation_capture_detail_code": "receipt_publication_interrupted",
                    }
                except Exception as exc:
                    if any(
                        isinstance(candidate, WorkspaceMutationSettlementError)
                        for candidate in iter_exception_tree(exc)
                    ):
                        workspace_settlement_failure = WorkspaceMutationSettlementError(
                            "Workspace mutation settlement could not be proven."
                        )
                        workspace_capture_payload = {
                            "workspace_mutation_capture_status": "failed",
                            "workspace_mutation_capture_detail_code": "mutation_settlement_unproven",
                        }
                    else:
                        workspace_capture_failure_detail = "receipt_publication_failed"
                        workspace_capture_payload = {
                            "workspace_mutation_capture_status": "failed",
                            "workspace_mutation_capture_detail_code": workspace_capture_failure_detail,
                        }
                else:
                    workspace_capture_payload = {
                        "workspace_mutation_capture_status": "pending",
                    }
            publication_snapshot = invocation_secret_scope.seal_for_publication()
            # The tool has returned an exact result. Sealed invocation/runner
            # evidence precedes terminal staging, so cancellation here must not
            # discard that result and turn a completed effect into ambiguity.
            publication_outcome = await await_shielded_task_outcome(
                asyncio.create_task(evidence.persist_sealed(publication_snapshot))
            )
            post_tool_cancellation_requests_consumed += (
                publication_outcome.cancellation_requests_consumed
            )
            if publication_outcome.cancellation is not None:
                observed_cancellation = await consume_post_tool_cancellation(
                    publication_outcome.cancellation
                )
                post_tool_cancellation = post_tool_cancellation or observed_cancellation
                if post_tool_cancellation is not None:
                    retain_cancellation_context(
                        post_tool_cancellation,
                        redactor=invocation_secret_scope.redactor,
                        tool_call_id=tool_call.id,
                    )
            if publication_outcome.error is not None:
                if post_tool_cancellation is not None:
                    _raise_preserved_post_tool_cancellation(
                        post_tool_cancellation,
                        publication_outcome.error,
                        restore_cancellation_requests=post_tool_cancellation_requests_consumed,
                    )
                raise publication_outcome.error

            hook_argument_projection = (
                tool_argument_publication.unavailable_argument_projection()
                if publish_arguments_as_unavailable
                else tool_argument_publication.finalized_argument_projection(
                    effective_tool_call.arguments,
                    redactor=publication_snapshot.redactor,
                    scope_finalized=not publication_snapshot.unsafe_output,
                )
            )
            argument_projection = (
                tool_argument_publication.unavailable_argument_projection()
                if publish_arguments_as_unavailable
                else tool_argument_publication.finalized_argument_projection(
                    model_arguments,
                    redactor=publication_snapshot.redactor,
                    scope_finalized=not publication_snapshot.unsafe_output,
                )
            )
            policy_denial = tool_context._policy_denial_for(registered_tool.tool)
            if policy_denial is not None:
                # Command-policy refusal occurs before the tool owns a publishable
                # result boundary. Command arguments can contain credentials even
                # when they were not resolved through the invocation secret
                # registry, so retain only explicit unavailability.
                argument_projection = tool_argument_publication.unavailable_argument_projection()
                hook_argument_projection = argument_projection
            published_terminal_event: Event | None = None
            projection_cancellation: asyncio.CancelledError | None = None
            if policy_denial is None:
                event_type = (
                    EventType.TOOL_CALL_FAILED if result.is_error else EventType.TOOL_CALL_COMPLETED
                )
                payload = {
                    "tool_call_id": tool_call.id,
                    "idempotency_key": idempotency_key,
                    **argument_projection.payload_fields(),
                    tool_argument_publication.ARGUMENTS_EXACT_FIELD: (
                        tool_argument_publication.argument_projection_is_exact(
                            argument_projection,
                            private_arguments=model_arguments,
                        )
                    ),
                    "result": result.model_dump(),
                    **effective_arguments_payload,
                    **execution_outcome.terminal_payload_fields(),
                    **workspace_capture_payload,
                    **identity_payload,
                }
                if approval_id is not None:
                    payload["approval_id"] = approval_id
                if input_id is not None:
                    payload["input_id"] = input_id
                result_event = Event(
                    type=event_type,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    tool_name=tool_call.name,
                    payload={},
                )
                result_event = result_event.model_copy(update={"payload": payload})
                (
                    result_event,
                    result,
                    size_failure,
                    initial_projection_cancellation,
                    initial_projection_requests,
                ) = await _await_post_tool_operation(
                    self.terminals.prepare_bounded_result(
                        event=result_event,
                        result=result,
                        registered_tool=registered_tool,
                        session=session,
                        registered_environment=registered_environment,
                        tool_call=effective_tool_call,
                        redactor=publication_snapshot.redactor,
                    ),
                    cancellation=post_tool_cancellation,
                    restore_cancellation_requests=post_tool_cancellation_requests_consumed,
                )
                if size_failure is not None:
                    execution_outcome = size_failure
                if initial_projection_cancellation is not None:
                    post_tool_cancellation_requests_consumed += initial_projection_requests
                    post_tool_cancellation = await consume_post_tool_cancellation(
                        initial_projection_cancellation
                    )
                result_event = _restore_targeted_tool_invocation_event_authority(
                    result_event,
                    effective_tool_call,
                    redactor=publication_snapshot.redactor,
                )
                if execution_outcome.publish_before_hooks and deferred_terminal_stager is None:
                    result_event, result = _prepare_tool_result_event(
                        event=result_event,
                        result=result,
                        redactor=output_redactor,
                        runtime_tool=registered_tool.tool,
                    )
                    if not output_redactor.has_same_registry(redactor):
                        result_event, result = _prepare_tool_result_event(
                            event=result_event,
                            result=result,
                            redactor=redactor,
                        )
                    (
                        result_event,
                        result,
                        _projection_failure,
                        projection_cancellation,
                        projection_requests,
                    ) = await _await_post_tool_operation(
                        self.terminals.project_result(
                            event=result_event,
                            result=result,
                            session=session,
                            registered_environment=registered_environment,
                            tool_call=effective_tool_call,
                            effect=registered_tool.effect,
                            redactor=redactor,
                        ),
                        cancellation=post_tool_cancellation,
                        restore_cancellation_requests=post_tool_cancellation_requests_consumed,
                    )
                    result_event = _restore_targeted_tool_invocation_event_authority(
                        result_event,
                        effective_tool_call,
                        redactor=publication_snapshot.redactor,
                    )
                    published_terminal_event = await _await_post_tool_operation(
                        self.terminals.emit(result_event),
                        cancellation=post_tool_cancellation,
                        restore_cancellation_requests=post_tool_cancellation_requests_consumed,
                    )
                    published_terminal_outcome = runtime_records.ToolCallOutcome(
                        call=replace(
                            effective_tool_call,
                            arguments=argument_projection.transcript_arguments(),
                            arguments_state=argument_projection.state,
                        ),
                        result=result,
                    )
                    # This terminal is already durable. Expose it before proxy
                    # telemetry so a later telemetry failure cannot erase the
                    # authoritative tool outcome from the public stream. Receipt
                    # events remain ordered immediately before that terminal.
                    for runner_event in runner_events:
                        yield runner_event, None
                    for workspace_event in workspace_events:
                        yield workspace_event, None
                    yield published_terminal_event, published_terminal_outcome
            proxy_events: list[Event] = []
            async for event in _iterate_post_tool_events(
                self._emit_proxy_authorization_events(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    tool_call=tool_call,
                    records=proxy_authorizations,
                    tool_round_identity=tool_round_identity,
                    execution_profile=execution_profile,
                    approval_id=approval_id,
                    input_id=input_id,
                    idempotency_key=idempotency_key,
                    redactor=redactor,
                    output_redactor=output_redactor,
                    quarantine_output=publish_arguments_as_unavailable,
                ),
                cancellation=post_tool_cancellation,
                restore_cancellation_requests=post_tool_cancellation_requests_consumed,
            ):
                if published_terminal_event is not None:
                    yield event, None
                else:
                    proxy_events.append(event)
            if projection_cancellation is not None:
                _raise_restored_post_tool_cancellation(
                    post_tool_cancellation or projection_cancellation,
                    restore_cancellation_requests=(
                        post_tool_cancellation_requests_consumed + projection_requests
                    ),
                )
            current_task = asyncio.current_task()
            tool_swallowed_cancellation = current_task is not None and current_task.cancelling() > 0
            if tool_swallowed_cancellation and await _await_post_tool_operation(
                self._session_control.is_interrupting(session.id),
                cancellation=post_tool_cancellation,
                restore_cancellation_requests=post_tool_cancellation_requests_consumed,
            ):
                raise SessionInterruptedByRequest(session.id)
            if policy_denial is not None:
                public_policy_denial_reason = output_redactor.redact_text(policy_denial.reason)
                terminal_events: list[tuple[Event, runtime_records.ToolCallOutcome | None]] = []
                terminal_recorded = False
                async for event in _iterate_post_tool_events(
                    self.terminals.emit_result(
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        tool_call=effective_tool_call,
                        event_type=EventType.TOOL_CALL_BLOCKED,
                        result=policy_denial.result,
                        extra_payload={
                            "idempotency_key": idempotency_key,
                            **workspace_capture_payload,
                            **policy_denial_payload_fields(
                                tool_name=effective_tool_call.name,
                                denied_by=policy_denial.denied_by,
                                decision=policy_denial.decision,
                                reason=public_policy_denial_reason,
                                metadata={},
                            ),
                        },
                        task_id=task_id,
                        execution_profile=execution_profile,
                        invocation_context=invocation_context,
                        tool_round_identity=tool_round_identity,
                        approval_id=approval_id,
                        input_id=input_id,
                        allow_modification=False,
                        redactor=redactor,
                        output_redactor=output_redactor,
                        argument_projection=argument_projection,
                        deferred_terminal_stager=effective_terminal_stager,
                        publication_snapshot=publication_snapshot,
                    ),
                    cancellation=post_tool_cancellation,
                    restore_cancellation_requests=post_tool_cancellation_requests_consumed,
                ):
                    if terminal_recorded:
                        yield event
                        continue
                    terminal_events.append(event)
                    if event[1] is None:
                        continue
                    terminal_recorded = True
                    for runner_event in runner_events:
                        yield runner_event, None
                    for workspace_event in workspace_events:
                        yield workspace_event, None
                    for proxy_event in proxy_events:
                        yield proxy_event, None
                    for terminal_event in terminal_events:
                        yield terminal_event
                    terminal_events.clear()
                if not terminal_recorded:
                    for runner_event in runner_events:
                        yield runner_event, None
                    for workspace_event in workspace_events:
                        yield workspace_event, None
                    for proxy_event in proxy_events:
                        yield proxy_event, None
                    for terminal_event in terminal_events:
                        yield terminal_event
                if post_tool_cancellation is not None:
                    _raise_restored_post_tool_cancellation(
                        post_tool_cancellation,
                        restore_cancellation_requests=post_tool_cancellation_requests_consumed,
                    )
                if workspace_settlement_failure is not None:
                    raise workspace_settlement_failure from None
                return
            if published_terminal_event is not None:
                hook_tool_call = _project_tool_call_for_hook(
                    effective_tool_call,
                    argument_projection=hook_argument_projection,
                    redactor=redactor,
                )
                async for hook_event, modified in _iterate_post_tool_events(
                    self.hooks.after_call(
                        session=session,
                        tool_event=published_terminal_event,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        tool_call=hook_tool_call,
                        result=result,
                        task_id=task_id,
                        execution_profile=execution_profile,
                        invocation_context=invocation_context,
                        redactor=redactor,
                        output_redactor=output_redactor,
                        allow_modification=False,
                        quarantine_output=not registered_tool.publish_arguments,
                    ),
                    cancellation=post_tool_cancellation,
                    restore_cancellation_requests=post_tool_cancellation_requests_consumed,
                ):
                    if modified is not None:
                        raise AssertionError(
                            "Observational after-tool hook modified terminal evidence."
                        )
                    yield hook_event, None
                if await _await_post_tool_operation(
                    self._session_control.is_interrupting(session.id),
                    cancellation=post_tool_cancellation,
                    restore_cancellation_requests=post_tool_cancellation_requests_consumed,
                ):
                    raise SessionInterruptedByRequest(session.id)
                if post_tool_cancellation is not None:
                    _raise_restored_post_tool_cancellation(
                        post_tool_cancellation,
                        restore_cancellation_requests=post_tool_cancellation_requests_consumed,
                    )
                if workspace_settlement_failure is not None:
                    raise workspace_settlement_failure from None
                return
            if result_event is None:
                raise AssertionError("Ordinary tool result event was not constructed.")
            terminal_events: list[tuple[Event, runtime_records.ToolCallOutcome | None]] = []
            terminal_recorded = False
            try:
                async for event in _iterate_post_tool_events(
                    self.terminals.publish_result(
                        event=result_event,
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        tool_call=effective_tool_call,
                        result=result,
                        task_id=task_id,
                        execution_profile=execution_profile,
                        invocation_context=invocation_context,
                        redactor=redactor,
                        output_redactor=output_redactor,
                        argument_projection=argument_projection,
                        hook_argument_projection=hook_argument_projection,
                        allow_modification=execution_outcome.allows_hook_modification,
                        publish_before_hooks=execution_outcome.publish_before_hooks,
                        deferred_terminal_stager=effective_terminal_stager,
                        publication_snapshot=publication_snapshot,
                        executed_runtime_tool=registered_tool.tool,
                    ),
                    cancellation=post_tool_cancellation,
                    restore_cancellation_requests=post_tool_cancellation_requests_consumed,
                ):
                    if terminal_recorded:
                        yield event
                        continue
                    terminal_events.append(event)
                    if event[1] is None:
                        continue
                    terminal_recorded = True
                    for runner_event in runner_events:
                        yield runner_event, None
                    for workspace_event in workspace_events:
                        yield workspace_event, None
                    for proxy_event in proxy_events:
                        yield proxy_event, None
                    for terminal_event in terminal_events:
                        yield terminal_event
                    terminal_events.clear()
            except ToolEffectReconciliationRequired:
                # These diagnostic events were already persisted after sealing
                # the invocation's secret scope. Uncertainty replaces the tool
                # terminal, not delivery of its preceding runner/proxy evidence.
                if not terminal_recorded:
                    for diagnostics in (runner_events, workspace_events, proxy_events):
                        for diagnostic_event in diagnostics:
                            yield diagnostic_event, None
                raise
            if not terminal_recorded:
                for runner_event in runner_events:
                    yield runner_event, None
                for workspace_event in workspace_events:
                    yield workspace_event, None
                for proxy_event in proxy_events:
                    yield proxy_event, None
                for terminal_event in terminal_events:
                    yield terminal_event
            if post_tool_cancellation is not None:
                _raise_restored_post_tool_cancellation(
                    post_tool_cancellation,
                    restore_cancellation_requests=post_tool_cancellation_requests_consumed,
                )
            if await _await_post_tool_operation(
                self._session_control.is_interrupting(session.id),
                cancellation=post_tool_cancellation,
                restore_cancellation_requests=post_tool_cancellation_requests_consumed,
            ):
                raise SessionInterruptedByRequest(session.id)
            if workspace_settlement_failure is not None:
                raise workspace_settlement_failure from None
        except BaseException as publication_failure:
            if dispatch.effect is not None:
                from cayu.runtime._tool_effect_conflicts import raise_after_tool_dispatch_audit

                await raise_after_tool_dispatch_audit(
                    self._session_store,
                    dispatch.effect,
                    publication_failure,
                    candidate_event_id=None if result_event is None else result_event.id,
                )
            raise

    async def _emit_proxy_authorization_events(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        tool_call: runtime_records.ToolCallRequest,
        records: list[invocation_secrets.ProxyAuthorizationRecord],
        tool_round_identity: ToolRoundIdentity,
        execution_profile: ExecutionProfileIdentity | None,
        approval_id: str | None,
        input_id: str | None,
        redactor: SecretRedactor,
        output_redactor: SecretRedactor,
        quarantine_output: bool = False,
        idempotency_key: str | None = None,
    ) -> AsyncIterator[Event]:
        identity_payload = copy_tool_round_identity(tool_round_identity).payload()
        for record in records:
            payload: dict[str, Any] = {
                "tool_call_id": tool_call.id,
                "destination": (
                    "unavailable"
                    if quarantine_output
                    else output_redactor.redact_text(record.destination)
                ),
                "credential": (
                    None
                    if quarantine_output or record.credential is None
                    else output_redactor.redact_text(record.credential.name)
                ),
                "action": (
                    None
                    if quarantine_output or record.action is None
                    else output_redactor.redact_text(record.action)
                ),
                "metadata": (
                    {}
                    if quarantine_output
                    else output_redactor.redact_json(copy_json_value(record.metadata, "metadata"))
                ),
                "allowed": record.result.allowed,
                "reason": (
                    None
                    if quarantine_output or record.result.reason is None
                    else output_redactor.redact_text(record.result.reason)
                ),
                "result_metadata": (
                    {}
                    if quarantine_output
                    else output_redactor.redact_json(
                        copy_json_value(record.result.metadata, "result_metadata")
                    )
                ),
                **identity_payload,
            }
            if idempotency_key is not None:
                payload["idempotency_key"] = idempotency_key
            if approval_id is not None:
                payload["approval_id"] = approval_id
            if input_id is not None:
                payload["input_id"] = input_id
            yield await self._event_writer.emit(
                prepare_runtime_event(
                    _event_with_tool_round_authority(
                        event_with_execution_profile_authority(
                            Event(
                                type=EventType.CREDENTIAL_PROXY_CHECKED,
                                session_id=session.id,
                                agent_name=registered_agent.spec.name,
                                environment_name=_environment_name(registered_environment),
                                tool_name=tool_call.name,
                                payload=payload,
                            ),
                            execution_profile,
                        ),
                        tool_round_identity,
                        *(
                            field_name
                            for field_name in ("approval_id", "input_id")
                            if field_name in payload
                        ),
                    ),
                    redactor=redactor,
                )
            )

    def detached_environment_work(self) -> set[asyncio.Future[Any]]:
        """Timed-out projections and abandoned secret resolutions still running."""

        return (
            self.terminals.detached_environment_work() | self.resources.detached_environment_work()
        )
