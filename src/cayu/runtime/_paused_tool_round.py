"""Shared round execution beneath typed approval and user-input continuations."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from cayu.approvals.tools import ToolApprovalDecision, ToolPolicyEvidence
from cayu.budgets.base import BudgetLimit
from cayu.events import Event, EventType
from cayu.execution_profiles import ExecutionProfileIdentity
from cayu.execution_units import ToolRoundIdentity
from cayu.runtime import _approval_support as approval_support
from cayu.runtime import _invocation_secrets as invocation_secrets
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _tool_execution as tool_execution
from cayu.runtime._auxiliary_invocation import AuxiliaryInvocationPolicy
from cayu.runtime._durable_tool_round import DurableToolRound
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime._invocation_lifecycle import InvocationContext
from cayu.runtime._tool_invocation.invocation import ToolInvocation
from cayu.runtime._tool_round_staging import _tool_terminal_payload_limits
from cayu.sessions.base import SessionStore
from cayu.sessions.records import Session
from cayu.tools.base import ToolEffect, ToolResult
from cayu.tools.exposure import ResolvedToolExposureAuthority
from cayu.tools.policy import ToolPolicyResult
from cayu.vaults.redaction import SecretRedactor


@dataclass(frozen=True, slots=True)
class ApprovalRoundPause:
    approval_id: str

    def authority(self) -> dict[str, str]:
        return {"approval_id": self.approval_id}

    def idempotency(self) -> dict[str, str]:
        return {"approval_id": self.approval_id}


@dataclass(frozen=True, slots=True)
class UserInputRoundPause:
    input_id: str

    def authority(self) -> dict[str, str]:
        return {"input_id": self.input_id}

    def idempotency(self) -> dict[str, str]:
        return {"pause_id": self.input_id}


@dataclass(frozen=True, slots=True)
class PausedToolRound:
    """Bind shared publication and dispatch to one admitted paused round.

    The typed continuation retains resolution validation, grant/answer events,
    checkpoint closure and failure recovery. This part does not authorize a gate.
    """

    session: Session
    agent: runtime_records.RegisteredAgentState
    environment: runtime_records.RegisteredEnvironment | None
    environment_name: str | None
    tool_exposure: ResolvedToolExposureAuthority | None
    secret_redactor: SecretRedactor
    profile: ExecutionProfileIdentity
    invocation_context: InvocationContext | None
    task_id: str | None
    identity: ToolRoundIdentity
    pause: ApprovalRoundPause | UserInputRoundPause
    invocation: ToolInvocation
    round: DurableToolRound
    redactor: SecretRedactor
    secret_resolution_scope: Literal["static", "dynamic", "unknown"]
    publish_arguments_as_unavailable: bool

    @classmethod
    async def prepare(
        cls,
        *,
        session_store: SessionStore,
        event_writer: RuntimeEventWriter,
        invocation: ToolInvocation,
        clock: Callable[[], datetime],
        session: Session,
        agent: runtime_records.RegisteredAgentState,
        environment: runtime_records.RegisteredEnvironment | None,
        environment_name: str | None,
        profile: ExecutionProfileIdentity,
        invocation_context: InvocationContext | None,
        identity: ToolRoundIdentity,
        task_id: str | None,
        tool_calls: list[runtime_records.ToolCallRequest],
        tool_exposure: ResolvedToolExposureAuthority | None,
        redactor: SecretRedactor,
        secret_redactor: SecretRedactor,
        secret_resolution_scope: Literal["static", "dynamic", "unknown"],
        pause: ApprovalRoundPause | UserInputRoundPause,
    ) -> PausedToolRound:
        defer_terminals = (len(tool_calls) > 1 and secret_resolution_scope != "static") or any(
            registered is not None
            and (registered.workspace_mutation or registered.effect is ToolEffect.EXTERNAL)
            for registered in (agent.executable_tool(call.name) for call in tool_calls)
        )
        round_owner = DurableToolRound.for_continuation(
            session=session,
            tool_round_identity=identity,
            session_store=session_store,
            event_writer=event_writer,
            registered_agent=agent,
            registered_environment=environment,
            environment_name=environment_name,
            tool_calls=tool_calls,
            task_id=task_id,
            execution_profile=profile,
            invocation_context=invocation_context,
            redactor=redactor,
            tool_exposure=tool_exposure,
            publication_governor=invocation.terminals.governor,
            clock=clock,
            emit_result=invocation.terminals.publish_result,
            emit_terminal=invocation.terminals.emit_staged,
            defer_terminals=defer_terminals,
            terminal_payload_limits=(
                await _tool_terminal_payload_limits(
                    agent,
                    tool_calls,
                    publication_governor=invocation.terminals.governor,
                    runtime_hooks=(
                        invocation.hooks.registrations
                        if invocation_context is None
                        else invocation_context.runtime_hooks
                    ),
                )
                if defer_terminals
                else None
            ),
            pause_authority=pause.authority(),
            idempotency_options=pause.idempotency(),
        )
        # The caller takes ownership before admit(), so its existing finally
        # also releases timing and reservations if admission fails.
        return cls(
            session=session,
            agent=agent,
            environment=environment,
            environment_name=environment_name,
            tool_exposure=tool_exposure,
            secret_redactor=secret_redactor,
            profile=profile,
            invocation_context=invocation_context,
            task_id=task_id,
            identity=identity,
            pause=pause,
            invocation=invocation,
            round=round_owner,
            redactor=redactor,
            secret_resolution_scope=secret_resolution_scope,
            publish_arguments_as_unavailable=len(tool_calls) > 1,
        )

    def publication_snapshot(self) -> invocation_secrets.InvocationPublicationSnapshot:
        return invocation_secrets.InvocationPublicationSnapshot(
            redactor=self.redactor,
            unsafe_output=False,
            secret_scope_incomplete=False,
        )

    def publish_result(
        self,
        *,
        event: Event,
        tool_call: runtime_records.ToolCallRequest,
        result: ToolResult,
    ) -> AsyncIterator[tuple[Event, runtime_records.ToolCallOutcome | None]]:
        return self.invocation.terminals.publish_result(
            event=event,
            session=self.session,
            registered_agent=self.agent,
            registered_environment=self.environment,
            tool_call=tool_call,
            result=result,
            task_id=self.task_id,
            execution_profile=self.profile,
            invocation_context=self.invocation_context,
            redactor=None if not self.round.defers_terminals else self.round.continuation_redactor,
            output_redactor=(
                None if not self.round.defers_terminals else self.round.continuation_redactor
            ),
            deferred_terminal_stager=(
                None if not self.round.defers_terminals else self.round.stage_terminal
            ),
            publication_snapshot=self.publication_snapshot(),
        )

    def execute(
        self,
        *,
        tool_call: runtime_records.ToolCallRequest,
        request_metadata: dict[str, Any],
        budget_limits: tuple[BudgetLimit, ...],
        auxiliary_invocation_policy: AuxiliaryInvocationPolicy,
        model_step: int | None,
        taint_labels: frozenset[str] | None,
        policy_result: ToolPolicyResult | None = None,
    ) -> AsyncIterator[tuple[Event, runtime_records.ToolCallOutcome | None]]:
        return self.round.timed_continuation_dispatch(
            self.invocation.execute(
                session=self.session,
                registered_agent=self.agent,
                registered_environment=self.environment,
                tool_call=tool_call,
                request_metadata=request_metadata,
                budget_limits=budget_limits,
                task_id=self.task_id,
                auxiliary_invocation_policy=auxiliary_invocation_policy,
                execution_profile=self.profile,
                invocation_context=self.invocation_context,
                check_policy=False,
                policy_result=policy_result,
                policy_output_secret_resolution_scope=self.secret_resolution_scope,
                tool_round_identity=self.identity,
                model_step=model_step,
                taint_labels=taint_labels,
                publish_arguments_as_unavailable=self.publish_arguments_as_unavailable,
                deferred_terminal_stager=(
                    None if not self.round.defers_terminals else self.round.stage_terminal
                ),
                deferred_terminal_capture_recorder=(
                    None if not self.round.defers_terminals else self.round.record_workspace_capture
                ),
                resolved_redactor_observer=(
                    None if not self.round.defers_terminals else self.round.record_redactor
                ),
                publication_snapshot_observer=self.round.record_publication_snapshot,
                rejoin_targeted_invocation=True,
                approval_id=(
                    self.pause.approval_id if isinstance(self.pause, ApprovalRoundPause) else None
                ),
                input_id=self.pause.input_id
                if isinstance(self.pause, UserInputRoundPause)
                else None,
            )
        )

    async def reject_policy(
        self,
        *,
        tool_call: runtime_records.ToolCallRequest,
        policy_evidence: ToolPolicyEvidence,
        requested_decision: ToolApprovalDecision | None = None,
        resolved_by_payload: dict[str, Any] | None = None,
        resolution_reason: str | None = None,
        resolution_metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[tuple[Event, runtime_records.ToolCallOutcome | None]]:
        """Close a call that lacks positive policy authority without dispatch."""
        approval_id = self.pause.approval_id if isinstance(self.pause, ApprovalRoundPause) else None
        input_id = self.pause.input_id if isinstance(self.pause, UserInputRoundPause) else None

        if (approval_id is None) == (input_id is None):
            raise TypeError("Exactly one approval or user-input identity is required.")
        if policy_evidence is ToolPolicyEvidence.UNEXPOSED:
            async for event, outcome in self.invocation.execute(
                session=self.session,
                registered_agent=self.agent,
                registered_environment=self.environment,
                tool_call=tool_call,
                request_metadata={},
                budget_limits=(),
                auxiliary_invocation_policy=None,
                task_id=self.task_id,
                execution_profile=self.profile,
                invocation_context=self.invocation_context,
                check_policy=False,
                emit_started=False,
                policy_evidence=policy_evidence,
                tool_exposure=self.tool_exposure,
                approval_id=approval_id,
                input_id=input_id,
                tool_round_identity=self.identity,
                deferred_terminal_stager=(
                    self.round.stage_terminal if self.round.defers_terminals else None
                ),
            ):
                yield event, outcome
            return
        evidence_payload: dict[str, Any]
        structured: dict[str, Any]
        if policy_evidence is ToolPolicyEvidence.AMBIGUOUS:
            event_type = EventType.TOOL_CALL_BLOCKED
            reason = (
                "Tool policy evaluation did not produce a durable decision; "
                "the call was not executed."
            )
            evidence_payload = {
                "decision": "ambiguous",
                "blocked_by": "policy_evaluation_ambiguous",
                "reason": reason,
            }
            structured = {
                "decision": "ambiguous",
                "blocked_by": "policy_evaluation_ambiguous",
            }
        elif policy_evidence is ToolPolicyEvidence.UNREGISTERED:
            event_type = EventType.TOOL_CALL_FAILED
            reason = f"Tool was not registered when the policy plan was recorded: {tool_call.name}"
            evidence_payload = {
                "registration_state": "unregistered_at_policy_plan",
            }
            structured = {
                "registration_state": "unregistered_at_policy_plan",
            }
        else:
            raise ValueError(
                "Non-authoritative closure requires ambiguous, unregistered, or unexposed evidence."
            )

        pause_payload = self.pause.authority()
        idempotency_options = self.pause.idempotency()
        if approval_id is not None:
            if requested_decision is not None:
                evidence_payload["requested_decision"] = requested_decision.value
                evidence_payload["resolution_reason"] = resolution_reason
                evidence_payload.update(
                    approval_support.bounded_resolution_metadata_payload(
                        {} if resolution_metadata is None else resolution_metadata,
                        redactor=self.secret_redactor,
                    )
                )
            evidence_payload["resolved_by"] = resolved_by_payload

        result = ToolResult(
            content=reason,
            structured={
                **self.identity.payload(),
                **pause_payload,
                "tool_call_id": tool_call.id,
                "tool_name": tool_call.name,
                **structured,
            },
            is_error=True,
        )
        idempotency_key = tool_execution.tool_idempotency_key(
            session_id=self.session.id,
            tool_round_id=self.identity.tool_round_id,
            tool_call_id=tool_call.id,
            **idempotency_options,
        )
        async for event, outcome in self.invocation.terminals.publish_result(
            event=Event(
                type=event_type,
                session_id=self.session.id,
                agent_name=self.agent.spec.name,
                environment_name=self.environment_name,
                tool_name=tool_call.name,
                payload={
                    **self.identity.payload(),
                    **pause_payload,
                    "tool_call_id": tool_call.id,
                    "idempotency_key": idempotency_key,
                    **evidence_payload,
                    "result": result.model_dump(),
                },
            ),
            session=self.session,
            registered_agent=self.agent,
            registered_environment=self.environment,
            tool_call=tool_call,
            result=result,
            task_id=self.task_id,
            execution_profile=self.profile,
            invocation_context=self.invocation_context,
            deferred_terminal_stager=(
                self.round.stage_terminal if self.round.defers_terminals else None
            ),
            publication_snapshot=self.publication_snapshot(),
        ):
            yield event, outcome
