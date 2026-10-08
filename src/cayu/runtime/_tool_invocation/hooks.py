"""Before/after tool hooks with invocation authority and secret-safe evidence."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from typing import Any

from cayu._validation import (
    copy_durable_json_object,
    copy_durable_json_value,
    require_durable_text,
)
from cayu.events import (
    Event,
    EventType,
    event_with_runtime_envelope_authority,
)
from cayu.execution_profiles import (
    ExecutionProfileIdentity,
    event_with_execution_profile_authority,
)
from cayu.observability.hooks import (
    AfterToolCallDecision,
    BeforeToolCallDecision,
    BeforeToolCallHookContext,
    RuntimeHookPhase,
    RuntimeHookRuntime,
    ToolCallHookContext,
    _runtime_hook_supports_phase,
)
from cayu.observability.hooks import _runtime_hook_event as _build_runtime_hook_event
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _tool_results as tool_results
from cayu.runtime._event_writer import RuntimeEventWriter, prepare_runtime_event
from cayu.runtime._invocation_lifecycle import InvocationContext
from cayu.runtime._tool_round_staging import (
    _is_policy_denial_event,
    _redact_tool_result_for_event,
)
from cayu.sessions.base import (
    INHERIT_INTERACTION,
    resolve_interaction_attribution,
)
from cayu.sessions.records import Session
from cayu.tools import _argument_publication as tool_argument_publication
from cayu.tools import _shared_artifact_results as shared_artifact_results
from cayu.tools import _web_access_results as web_access_results
from cayu.tools.base import (
    ToolResult,
)
from cayu.vaults.redaction import SecretRedactor

from .context import _environment_name


@dataclass
class _BeforeToolCallResolution:
    arguments: dict[str, Any]
    short_circuit_result: ToolResult | None = None
    block_reason: str | None = None


def _private_argument_short_circuit_result(result: ToolResult) -> ToolResult:
    """Project one hook-controlled result without argument-derived content."""

    return ToolResult(
        content="Tool execution was short-circuited by a before_tool_call hook.",
        structured={"outcome": "short_circuited"},
        is_error=result.is_error,
    )


def _resolve_before_tool_call_decision(
    decision: BeforeToolCallDecision | None,
    resolution: _BeforeToolCallResolution,
    *,
    redactor: SecretRedactor,
) -> bool:
    if decision is None:
        return False
    if type(decision) is not BeforeToolCallDecision:
        raise TypeError("before_tool_call must return a BeforeToolCallDecision or None.")
    if decision.action == "proceed":
        return False
    if decision.action == "proceed_modified":
        modified_arguments = decision.modified_arguments
        if modified_arguments is None:
            raise TypeError("A proceed_modified decision must carry modified_arguments.")
        copied_arguments = copy_durable_json_value(
            modified_arguments,
            "modified_arguments",
        )
        if type(copied_arguments) is not dict:
            raise TypeError("A proceed_modified decision must carry object arguments.")
        resolution.arguments = copied_arguments
        return False
    if decision.action == "short_circuit":
        synthetic = decision.synthetic_result
        if synthetic is None:
            raise TypeError("A short_circuit decision must carry a synthetic_result.")
        resolution.short_circuit_result = _prepare_hook_authored_tool_result(
            synthetic,
            redactor=redactor,
        )
        return True
    if decision.action != "block":
        raise ValueError("Unsupported before_tool_call decision action.")
    reason = decision.block_reason
    if reason is None:
        raise TypeError("A block decision must carry a block_reason.")
    resolution.block_reason = require_durable_text(reason, "block_reason")
    return True


def _resolve_after_tool_call_decision(
    decision: AfterToolCallDecision | None,
    *,
    redactor: SecretRedactor,
) -> ToolResult | None:
    if decision is None:
        return None
    if type(decision) is not AfterToolCallDecision:
        raise TypeError("after_tool_call must return an AfterToolCallDecision or None.")
    if decision.action == "modify":
        modified = decision.modified_result
        if modified is None:
            raise TypeError("An after_tool_call modify decision must carry a modified_result.")
        return _prepare_hook_authored_tool_result(modified, redactor=redactor)
    if decision.action == "pass_through":
        if decision.modified_result is not None:
            raise TypeError("A pass_through decision must not carry a modified_result.")
        return None
    raise ValueError("Unsupported after_tool_call decision action.")


def _prepare_hook_authored_tool_result(
    result: ToolResult,
    *,
    redactor: SecretRedactor,
) -> ToolResult:
    """Detach untrusted hook output before the next runtime-owned await."""

    validated = tool_results.normalize_tool_result(tool_results.validate_tool_result(result))
    sanitized = ToolResult(
        content=validated.content,
        structured=tool_results.strip_untrusted_runtime_tool_result_control_authority(
            validated.structured
        ),
        artifacts=tool_results.strip_runtime_tool_result_projection_authority(validated.artifacts),
        is_error=validated.is_error,
    )
    return tool_results.redact_tool_result(sanitized, redactor)


def _runtime_hook_event(
    *,
    event_type: EventType,
    hook_name: str,
    scope: str,
    phase: RuntimeHookPhase,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    registered_environment: runtime_records.RegisteredEnvironment | None,
    terminal_event: Event,
    payload: dict[str, Any],
    execution_profile: ExecutionProfileIdentity | None = None,
) -> Event:
    event = _build_runtime_hook_event(
        event_type=event_type,
        hook_name=hook_name,
        scope=scope,
        phase=phase,
        session=session,
        terminal_event=terminal_event,
        agent_name=registered_agent.spec.name,
        environment_name=_environment_name(registered_environment),
        payload=payload,
    )
    # Tool hooks belong to their tool event's interaction. A result event can
    # still be unpublished here, so fall back to the active tool-round owner.
    # Keep this scoped: terminal hooks use deterministic replay identities and
    # must preserve the terminal event's explicit attribution instead.
    interaction_id = terminal_event.interaction_id
    if interaction_id is None:
        interaction_id = resolve_interaction_attribution(session.id, INHERIT_INTERACTION)
    if interaction_id is not None:
        event = event_with_runtime_envelope_authority(
            event.model_copy(update={"interaction_id": interaction_id}),
            "interaction_id",
        )
    return event_with_execution_profile_authority(
        event,
        execution_profile,
    )


def _redact_event_for_invocation(
    event: Event,
    *,
    redactor: SecretRedactor,
) -> Event:
    """Scrub adapter/invocation secrets before the app-level writer sees them."""

    return prepare_runtime_event(event, redactor=redactor)


def _project_tool_call_for_hook(
    tool_call: runtime_records.ToolCallRequest,
    *,
    argument_projection: tool_argument_publication.ToolArgumentProjection,
    redactor: SecretRedactor,
) -> runtime_records.ToolCallRequest:
    """Expose effective arguments only after the invocation scope is complete."""

    if argument_projection.state == "unavailable":
        arguments: dict[str, Any] = {}
    else:
        projected = redactor.redact_json(argument_projection.transcript_arguments())
        if type(projected) is not dict:
            raise AssertionError("Hook argument projection returned a non-object.")
        arguments = projected
    return replace(tool_call, arguments=arguments)


def _hook_failure_payload(
    exc: Exception,
    *,
    redactor: SecretRedactor,
) -> dict[str, Any]:
    diagnostic = tool_results.exception_diagnostic(
        exc,
        empty_message="runtime hook failed",
        nonportable_message="Runtime hook failed with a non-portable diagnostic.",
        redactor=redactor,
    )
    return copy_durable_json_object(diagnostic.payload_fields(), "hook_failure")


def _hook_actions_payload(
    context: BeforeToolCallHookContext | ToolCallHookContext,
    *,
    redactor: SecretRedactor,
) -> dict[str, Any]:
    try:
        actions = copy_durable_json_value(context.actions, "hook_actions")
        actions = redactor.redact_json(actions)
        actions = copy_durable_json_value(actions, "hook_actions")
    except Exception:
        return {"actions": [], "actions_omitted": True}
    if type(actions) is not list:
        return {"actions": [], "actions_omitted": True}
    return {"actions": actions}


class ToolInvocationHooks:
    """Run hook phases independently of tool dispatch and terminal persistence."""

    def __init__(
        self,
        *,
        event_writer: RuntimeEventWriter,
        hook_runtime: RuntimeHookRuntime,
        runtime_hooks: tuple[runtime_records.RegisteredRuntimeHook, ...],
    ) -> None:
        self._event_writer = event_writer
        self._hook_runtime = hook_runtime
        self.registrations = runtime_hooks

    async def before_call(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        tool_call: runtime_records.ToolCallRequest,
        anchor_event: Event,
        task_id: str | None,
        resolution: _BeforeToolCallResolution,
        redactor: SecretRedactor,
        output_redactor: SecretRedactor,
        execution_profile: ExecutionProfileIdentity | None,
        invocation_context: InvocationContext | None = None,
        quarantine_output: bool = False,
    ) -> AsyncIterator[Event]:
        if invocation_context is not None and (
            invocation_context.binding.session_id != session.id
            or invocation_context.registered_agent is not registered_agent
            or invocation_context.registered_environment is not registered_environment
            or invocation_context.profile is not execution_profile
        ):
            raise RuntimeError("Before-tool hooks lost frozen invocation authority.")
        runtime_hooks = (
            self.registrations if invocation_context is None else invocation_context.runtime_hooks
        )
        for hooks, scope in (
            (runtime_hooks, "app"),
            (registered_agent.runtime_hooks, "agent"),
        ):
            for registered_hook in hooks:
                hook = registered_hook.hook
                if not _runtime_hook_supports_phase(
                    hook=hook,
                    phase=RuntimeHookPhase.BEFORE_TOOL_CALL,
                ):
                    continue
                hook_name = registered_hook.name
                yield await self._event_writer.emit(
                    _redact_event_for_invocation(
                        _runtime_hook_event(
                            event_type=EventType.HOOK_STARTED,
                            hook_name=hook_name,
                            scope=scope,
                            phase=RuntimeHookPhase.BEFORE_TOOL_CALL,
                            session=session,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            terminal_event=anchor_event,
                            execution_profile=execution_profile,
                            payload={
                                "tool_name": tool_call.name,
                                "tool_call_id": tool_call.id,
                            },
                        ),
                        redactor=redactor,
                    )
                )
                context = BeforeToolCallHookContext(
                    runtime=self._hook_runtime,
                    hook_name=hook_name,
                    phase=RuntimeHookPhase.BEFORE_TOOL_CALL,
                    session=session,
                    tool_name=tool_call.name,
                    tool_call_id=tool_call.id,
                    arguments=resolution.arguments,
                    task_id=task_id,
                    publication_actions_allowed=not quarantine_output,
                    execution_profile=execution_profile,
                )
                hook_failure_payload: dict[str, Any] | None = None
                try:
                    decision = await hook.before_tool_call(context)
                    try:
                        stop = _resolve_before_tool_call_decision(
                            decision,
                            resolution,
                            redactor=output_redactor,
                        )
                    finally:
                        # Hook-owned objects can retain application secrets. Do
                        # not carry the raw decision across durable publication.
                        decision = None
                except Exception as exc:
                    hook_failure_payload = (
                        {
                            "error_type": "runtime_hook_failure",
                            "actions": [],
                            "actions_omitted": True,
                        }
                        if quarantine_output
                        else {
                            **_hook_failure_payload(
                                exc,
                                redactor=output_redactor,
                            ),
                            **_hook_actions_payload(
                                context,
                                redactor=output_redactor,
                            ),
                        }
                    )
                # Publish only after the raw hook exception has left the active
                # handler so a store failure cannot inherit its traceback.
                if hook_failure_payload is not None:
                    yield await self._event_writer.emit(
                        _redact_event_for_invocation(
                            _runtime_hook_event(
                                event_type=EventType.HOOK_FAILED,
                                hook_name=hook_name,
                                scope=scope,
                                phase=RuntimeHookPhase.BEFORE_TOOL_CALL,
                                session=session,
                                registered_agent=registered_agent,
                                registered_environment=registered_environment,
                                terminal_event=anchor_event,
                                execution_profile=execution_profile,
                                payload={
                                    "tool_name": tool_call.name,
                                    "tool_call_id": tool_call.id,
                                    **hook_failure_payload,
                                },
                            ),
                            redactor=redactor,
                        )
                    )
                    continue
                yield await self._event_writer.emit(
                    _redact_event_for_invocation(
                        _runtime_hook_event(
                            event_type=EventType.HOOK_COMPLETED,
                            hook_name=hook_name,
                            scope=scope,
                            phase=RuntimeHookPhase.BEFORE_TOOL_CALL,
                            session=session,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            terminal_event=anchor_event,
                            execution_profile=execution_profile,
                            payload={
                                "tool_name": tool_call.name,
                                "tool_call_id": tool_call.id,
                                **(
                                    {"actions": [], "actions_omitted": True}
                                    if quarantine_output
                                    else _hook_actions_payload(
                                        context,
                                        redactor=output_redactor,
                                    )
                                ),
                            },
                        ),
                        redactor=redactor,
                    )
                )
                if stop:
                    return

    async def after_call(
        self,
        *,
        session: Session,
        tool_event: Event,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        tool_call: runtime_records.ToolCallRequest,
        result: ToolResult,
        task_id: str | None,
        redactor: SecretRedactor,
        output_redactor: SecretRedactor,
        execution_profile: ExecutionProfileIdentity | None = None,
        invocation_context: InvocationContext | None = None,
        allow_modification: bool = False,
        quarantine_output: bool = False,
    ) -> AsyncIterator[tuple[Event, ToolResult | None]]:
        if invocation_context is not None and (
            invocation_context.binding.session_id != session.id
            or invocation_context.registered_agent is not registered_agent
            or invocation_context.registered_environment is not registered_environment
            or invocation_context.profile is not execution_profile
        ):
            raise RuntimeError("After-tool hooks lost frozen invocation authority.")
        current_result = result
        runtime_hooks = (
            self.registrations if invocation_context is None else invocation_context.runtime_hooks
        )
        for hooks, scope in (
            (runtime_hooks, "app"),
            (registered_agent.runtime_hooks, "agent"),
        ):
            async for hook_event, modified in self._run_scoped(
                session=session,
                tool_event=tool_event,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                tool_call=tool_call,
                result=current_result,
                task_id=task_id,
                execution_profile=execution_profile,
                hooks=hooks,
                scope=scope,
                redactor=redactor,
                output_redactor=output_redactor,
                allow_modification=allow_modification,
                quarantine_output=quarantine_output,
                attested_result=result,
            ):
                yield hook_event, modified
                if modified is not None:
                    current_result = modified

    async def _run_scoped(
        self,
        *,
        session: Session,
        tool_event: Event,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        tool_call: runtime_records.ToolCallRequest,
        result: ToolResult,
        task_id: str | None,
        hooks: tuple[runtime_records.RegisteredRuntimeHook, ...],
        scope: str,
        redactor: SecretRedactor,
        output_redactor: SecretRedactor,
        execution_profile: ExecutionProfileIdentity | None = None,
        allow_modification: bool = False,
        quarantine_output: bool = False,
        attested_result: ToolResult | None = None,
    ) -> AsyncIterator[tuple[Event, ToolResult | None]]:
        current_result = result
        for registered_hook in hooks:
            hook = registered_hook.hook
            if not _runtime_hook_supports_phase(
                hook=hook,
                phase=RuntimeHookPhase.AFTER_TOOL_CALL,
            ):
                continue
            hook_name = registered_hook.name
            yield (
                await self._event_writer.emit(
                    _redact_event_for_invocation(
                        _runtime_hook_event(
                            event_type=EventType.HOOK_STARTED,
                            hook_name=hook_name,
                            scope=scope,
                            phase=RuntimeHookPhase.AFTER_TOOL_CALL,
                            session=session,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            terminal_event=tool_event,
                            execution_profile=execution_profile,
                            payload={
                                "tool_name": tool_call.name,
                                "tool_call_id": tool_call.id,
                            },
                        ),
                        redactor=redactor,
                    )
                ),
                None,
            )
            context = ToolCallHookContext(
                runtime=self._hook_runtime,
                hook_name=hook_name,
                phase=RuntimeHookPhase.AFTER_TOOL_CALL,
                session=session,
                tool_event=tool_event,
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                arguments=output_redactor.redact_json(tool_call.arguments),
                result=(
                    current_result
                    if _is_policy_denial_event(tool_event) and not allow_modification
                    else _redact_tool_result_for_event(
                        event=tool_event,
                        result=current_result,
                        redactor=output_redactor,
                    )
                ),
                task_id=task_id,
                publication_actions_allowed=not quarantine_output,
                execution_profile=execution_profile,
            )
            hook_failure_payload: dict[str, Any] | None = None
            try:
                decision = await hook.after_tool_call(context)
                try:
                    resolved = _resolve_after_tool_call_decision(
                        decision,
                        redactor=output_redactor,
                    )
                finally:
                    # The copied, sanitized resolution is the only hook result
                    # permitted to cross the following durable event await.
                    decision = None
                modified = resolved if allow_modification else None
            except Exception as exc:
                hook_failure_payload = (
                    {
                        "error_type": "runtime_hook_failure",
                        "actions": [],
                        "actions_omitted": True,
                    }
                    if quarantine_output
                    else {
                        **_hook_failure_payload(
                            exc,
                            redactor=output_redactor,
                        ),
                        **_hook_actions_payload(
                            context,
                            redactor=output_redactor,
                        ),
                    }
                )
            # Publish only after the raw hook exception has left the active
            # handler so a store failure cannot inherit its traceback.
            if hook_failure_payload is not None:
                yield (
                    await self._event_writer.emit(
                        _redact_event_for_invocation(
                            _runtime_hook_event(
                                event_type=EventType.HOOK_FAILED,
                                hook_name=hook_name,
                                scope=scope,
                                phase=RuntimeHookPhase.AFTER_TOOL_CALL,
                                session=session,
                                registered_agent=registered_agent,
                                registered_environment=registered_environment,
                                terminal_event=tool_event,
                                execution_profile=execution_profile,
                                payload={
                                    "tool_name": tool_call.name,
                                    "tool_call_id": tool_call.id,
                                    **hook_failure_payload,
                                },
                            ),
                            redactor=redactor,
                        )
                    ),
                    None,
                )
                continue
            if modified is not None:
                if attested_result is not None:
                    modified = web_access_results.preserve_attested_controls_across_hook(
                        tool_event,
                        original=attested_result,
                        replacement=modified,
                    )
                    modified = shared_artifact_results.preserve_attested_controls_across_hook(
                        tool_event,
                        original=attested_result,
                        replacement=modified,
                    )
                current_result = modified
            yield (
                await self._event_writer.emit(
                    _redact_event_for_invocation(
                        _runtime_hook_event(
                            event_type=EventType.HOOK_COMPLETED,
                            hook_name=hook_name,
                            scope=scope,
                            phase=RuntimeHookPhase.AFTER_TOOL_CALL,
                            session=session,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            terminal_event=tool_event,
                            execution_profile=execution_profile,
                            payload={
                                "tool_name": tool_call.name,
                                "tool_call_id": tool_call.id,
                                **(
                                    {"actions": [], "actions_omitted": True}
                                    if quarantine_output
                                    else _hook_actions_payload(
                                        context,
                                        redactor=output_redactor,
                                    )
                                ),
                            },
                        ),
                        redactor=redactor,
                    )
                ),
                modified,
            )
