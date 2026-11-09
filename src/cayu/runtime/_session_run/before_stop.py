"""Before-stop loop policy selection and events."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from cayu._validation import copy_json_value, require_clean_nonblank
from cayu.events import Event
from cayu.execution_profiles import ExecutionProfileIdentity
from cayu.messages import Message
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime._diagnostics import exception_diagnostic
from cayu.runtime._durable_tool_round import _environment_name
from cayu.runtime._invocation_lifecycle import InvocationContext
from cayu.runtime._loop_policy_continuations import (
    before_stop_continuation_indices,
    before_stop_policy_key,
)
from cayu.runtime._run_limits import SessionUsageTracker
from cayu.runtime._session_control import ActiveSessionRun
from cayu.runtime.loop_policies import (
    BeforeStopAction,
    BeforeStopContext,
    BeforeStopDecision,
    LoopPolicy,
    copy_before_stop_decision,
    validate_loop_policies,
)
from cayu.runtime.model_steps import (
    AssistantStepResult,
    StepClassification,
    classify_assistant_step,
)
from cayu.runtime.stop_policy import StopDecision, StopLimit
from cayu.sessions.records import Session

if TYPE_CHECKING:
    from cayu.runtime._session_engine import SessionEngine


def _loop_policy_supports_before_stop(policy: LoopPolicy) -> bool:
    policy_method = type(policy).before_stop
    default_method = LoopPolicy.before_stop
    return policy_method is not default_method


@dataclass(frozen=True)
class _BeforeStopPolicySelection:
    decision: BeforeStopDecision
    policy_key: str


def _before_stop_policy_event(
    *,
    event_type: str,
    policy_name: str,
    scope: str,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    registered_environment: runtime_records.RegisteredEnvironment | None,
    step: int,
    classification: StepClassification,
    payload: dict[str, Any],
) -> Event:
    event_payload = {
        "policy": policy_name,
        "scope": scope,
        "step": step,
        "classification": classification.payload(),
        **copy_json_value(payload, "payload"),
    }
    return Event(
        type=event_type,
        session_id=session.id,
        agent_name=registered_agent.spec.name,
        environment_name=_environment_name(registered_environment),
        payload=event_payload,
    )


class BeforeStopPolicies:
    """Before-stop policies and the model-step limit stop for ``SessionEngine``."""

    async def _stop_session_for_model_step_limit(
        self: SessionEngine,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_name: str | None,
        messages: list[Message],
        step: int,
        max_steps: int | None,
        run_started_at: float,
        turn_usage_tracker: SessionUsageTracker,
        active_run: ActiveSessionRun[SessionUsageTracker] | None,
        execution_profile: ExecutionProfileIdentity | None,
        invocation_context: InvocationContext | None = None,
    ) -> AsyncGenerator[Event, None]:
        if max_steps is None:
            raise RuntimeError("An unlimited invocation cannot exhaust model steps.")
        usage_summary = await self._run_limit_controller.session_usage_summary(session.id)
        decision = StopDecision(
            limit=StopLimit.MODEL_STEPS,
            maximum=max_steps,
            actual=step,
            message=(
                "Run limit reached: the configured model-step allowance was "
                f"exhausted ({step} >= {max_steps})."
            ),
        )
        async for event in self._session_finalization.stop_session_for_limit_reached(
            session=session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            environment_name=environment_name,
            decision=decision,
            usage_summary=usage_summary,
            cost_summary=None,
            messages=messages,
            tool_calls=[],
            completed_tool_outcomes=[],
            run_started_at=run_started_at,
            turn_usage_tracker=turn_usage_tracker,
            active_run=active_run,
            execution_profile=execution_profile,
            invocation_context=invocation_context,
            reconcile_transition_cancellation=False,
        ):
            yield event

    async def _run_before_stop_policies(
        self: SessionEngine,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        step_result: AssistantStepResult,
        step: int,
        max_steps: int | None,
        request_metadata: dict[str, Any],
        request_loop_policies: tuple[LoopPolicy, ...],
        invocation_context: InvocationContext | None = None,
    ) -> AsyncGenerator[tuple[Event, _BeforeStopPolicySelection | None], None]:
        if invocation_context is not None and (
            invocation_context.binding.session_id != session.id
            or registered_agent is not invocation_context.registered_agent
            or registered_environment is not invocation_context.registered_environment
            or len(invocation_context.request_loop_policies) != len(request_loop_policies)
            or any(
                frozen is not supplied
                for frozen, supplied in zip(
                    invocation_context.request_loop_policies,
                    request_loop_policies,
                    strict=True,
                )
            )
        ):
            raise RuntimeError("Loop policy substituted frozen invocation authority.")
        classification = classify_assistant_step(step_result)
        policy_groups = (
            (
                "app",
                self._loop_policies
                if invocation_context is None
                else invocation_context.loop_policies,
            ),
            ("agent", registered_agent.loop_policies),
            (
                "request",
                validate_loop_policies(
                    request_loop_policies
                    if invocation_context is None
                    else invocation_context.request_loop_policies,
                    field_name="request_loop_policies",
                ),
            ),
        )
        snapshot = None
        checkpoint = None
        for scope, policies in policy_groups:
            for position, policy in enumerate(policies):
                if not _loop_policy_supports_before_stop(policy):
                    continue
                if snapshot is None:
                    snapshot = await self.session_store.load_transcript_snapshot(session.id)
                    checkpoint = await self.session_store.load_checkpoint(session.id)
                policy_key = before_stop_policy_key(policy, scope=scope, position=position)
                policy_name = require_clean_nonblank(policy.name, "loop_policy.name")
                yield (
                    await self._event_writer.emit(
                        _before_stop_policy_event(
                            event_type="custom.loop.before_stop.started",
                            policy_name=policy_name,
                            scope=scope,
                            session=session,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            step=step,
                            classification=classification,
                            payload={},
                        )
                    ),
                    None,
                )
                context = BeforeStopContext(
                    session=session,
                    step_result=step_result,
                    classification=classification,
                    step=step,
                    max_steps=max_steps,
                    metadata=request_metadata,
                    messages=[record.message for record in snapshot.records],
                    continuation_message_indices=before_stop_continuation_indices(
                        checkpoint,
                        snapshot=snapshot,
                        session_id=session.id,
                        policy_key=policy_key,
                    ),
                )
                try:
                    decision = copy_before_stop_decision(await policy.before_stop(context))
                except Exception as exc:
                    diagnostic = exception_diagnostic(
                        exc,
                        empty_message="loop policy failed",
                        nonportable_message=("Loop policy failed with a non-portable diagnostic."),
                        redactor=self._secret_redactor,
                    )
                    yield (
                        await self._event_writer.emit(
                            _before_stop_policy_event(
                                event_type="custom.loop.before_stop.failed",
                                policy_name=policy_name,
                                scope=scope,
                                session=session,
                                registered_agent=registered_agent,
                                registered_environment=registered_environment,
                                step=step,
                                classification=classification,
                                payload=diagnostic.payload_fields(),
                            )
                        ),
                        None,
                    )
                    raise
                yield (
                    await self._event_writer.emit(
                        _before_stop_policy_event(
                            event_type="custom.loop.before_stop.completed",
                            policy_name=policy_name,
                            scope=scope,
                            session=session,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            step=step,
                            classification=classification,
                            payload={
                                "action": decision.action.value,
                                "reason": decision.reason,
                                "metadata": copy_json_value(decision.metadata, "metadata"),
                            },
                        )
                    ),
                    None,
                )
                if decision.action != BeforeStopAction.COMPLETE:
                    yield (
                        await self._event_writer.emit(
                            _before_stop_policy_event(
                                event_type="custom.loop.before_stop.selected",
                                policy_name=policy_name,
                                scope=scope,
                                session=session,
                                registered_agent=registered_agent,
                                registered_environment=registered_environment,
                                step=step,
                                classification=classification,
                                payload={
                                    "action": decision.action.value,
                                    "reason": decision.reason,
                                    "metadata": copy_json_value(
                                        decision.metadata,
                                        "metadata",
                                    ),
                                },
                            )
                        ),
                        _BeforeStopPolicySelection(decision=decision, policy_key=policy_key),
                    )
                    return
