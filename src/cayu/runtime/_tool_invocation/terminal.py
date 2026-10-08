"""Bounded tool-result projection, hooks and durable terminal publication."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import suppress
from dataclasses import replace
from datetime import datetime
from typing import Any, cast

from cayu._task_wait import (
    await_shielded_task_outcome,
    unexpected_child_cancellation_error,
)
from cayu._validation import (
    DurableValueError,
    canonical_durable_json_bytes,
    extract_durable_value_error,
)
from cayu.artifacts.settlement import ArtifactWriteSettlementObserver
from cayu.events import (
    Event,
    EventType,
    copy_event,
    event_nested_payload_authority_is_runtime_generated,
    event_with_runtime_nested_payload_authority,
    validate_event_envelope,
)
from cayu.execution_profiles import (
    ExecutionProfileIdentity,
    event_with_execution_profile_authority,
)
from cayu.execution_units import ToolRoundIdentity, copy_tool_round_identity
from cayu.observability.hooks import (
    RuntimeHookPhase,
    _runtime_hook_supports_phase,
)
from cayu.runtime import _invocation_secrets as invocation_secrets
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _tool_execution as tool_execution
from cayu.runtime import _tool_results as tool_results
from cayu.runtime._event_writer import RuntimeEventWriter, prepare_runtime_event
from cayu.runtime._invocation_lifecycle import InvocationContext
from cayu.runtime._phase_timing import timed_phase
from cayu.runtime._tool_effect_state import (
    ToolEffectRecord,
    ToolEffectStateOwner,
    ToolEffectTerminal,
)
from cayu.runtime._tool_round_staging import (
    _durable_payload_utf8_size,
    _event_with_tool_round_authority,
    _prepare_tool_result_event,
    _terminal_publication_work_estimate,
    _validate_and_synchronize_tool_result_event,
)
from cayu.sessions.base import (
    SessionStore,
)
from cayu.sessions.records import Session
from cayu.tools import _argument_publication as tool_argument_publication
from cayu.tools import _shared_artifact_results as shared_artifact_results
from cayu.tools import _terminal_controls as tool_terminal_controls
from cayu.tools import _web_access_results as web_access_results
from cayu.tools.base import (
    ToolEffect,
    ToolResult,
)
from cayu.tools.catalogue import ToolExecutionContract
from cayu.tools.result_projection import (
    _TOOL_RESULT_PROJECTION_PROVENANCE_PATH,
    ToolResultProjectionPolicy,
    ToolResultProjectionRequest,
    _last_reconciliation_candidate,
    projection_failure,
    safe_projection_failure_type,
    validate_tool_result_projection,
)
from cayu.tools.terminal_publication import (
    TOOL_TERMINAL_PUBLICATION_SLICE_BYTES,
    ToolTerminalPublicationGovernor,
    ToolTerminalPublicationMetricsSnapshot,
)
from cayu.vaults.redaction import SecretRedactor

from .cancellation import _raise_restored_post_tool_cancellation
from .context import (
    _artifact_store,
    _environment_name,
    _event_with_targeted_tool_invocation_authority,
    _published_argument_presence,
    _restore_targeted_tool_invocation_event_authority,
)
from .hooks import ToolInvocationHooks, _project_tool_call_for_hook

_TOOL_RESULT_PROJECTION_TIMEOUT_SECONDS = 30.0


def _bounded_tool_failure_event(
    event: Event, failure: tool_execution.ToolExecutionOutcome
) -> Event:
    payload = dict(event.payload)
    payload.pop("tool_result_projection", None)
    payload["result"] = failure.result.model_dump(mode="json")
    payload.update(failure.terminal_payload_fields())
    failed = event.model_copy(update={"type": EventType.TOOL_CALL_FAILED, "payload": payload})
    try:
        copied = copy_event(failed)
        validate_event_envelope(copied)
        return copied
    except Exception as exc:
        limit_error = extract_durable_value_error(exc)
        if limit_error is None or limit_error.dimension is None:
            raise
    # The effect intent retains argument authority. This terminal projection is
    # explicitly unavailable, never a truncated or different invocation.
    payload.pop("effective_arguments", None)
    payload.pop(tool_argument_publication.ARGUMENTS_FIELD, None)
    payload.update(tool_argument_publication.unavailable_argument_projection().payload_fields())
    payload[tool_argument_publication.ARGUMENTS_EXACT_FIELD] = False
    copied = copy_event(failed.model_copy(update={"payload": payload}))
    validate_event_envelope(copied)
    return copied


def _consume_projection_task_outcome(task: asyncio.Task[Any]) -> None:
    """Observe a timed-out policy task after requesting cooperative cancellation."""

    with suppress(BaseException):
        task.result()


DeferredTerminalStager = Callable[
    [
        Event,
        runtime_records.ToolCallOutcome,
        bool,
        bool,
        invocation_secrets.InvocationPublicationSnapshot,
    ],
    Awaitable[Event],
]


DeferredTerminalFinalizer = Callable[[Event], Awaitable[Event]]


DeferredTerminalCaptureRecorder = Callable[[Event], Awaitable[Event]]


TerminalEventEmitter = Callable[[Event], Awaitable[Event]]


class ToolTerminalPublisher:
    """Publish live or recovered outcomes through the same terminal boundary."""

    def __init__(
        self,
        *,
        event_writer: RuntimeEventWriter,
        hooks: ToolInvocationHooks,
        secret_redactor: SecretRedactor,
        projection_policy: ToolResultProjectionPolicy | None,
        clock: Callable[[], datetime],
    ) -> None:
        self._event_writer = event_writer
        self._hooks = hooks
        self._secret_redactor = secret_redactor
        self._tool_result_projection_policy = projection_policy
        self._clock = clock
        self.governor = ToolTerminalPublicationGovernor()
        self._detached_projections: set[asyncio.Task[Any]] = set()

    async def publish_unexecuted(
        self,
        *,
        tool_call_id: str,
        event: Event,
        outcome: runtime_records.ToolCallOutcome,
        snapshot: invocation_secrets.InvocationPublicationSnapshot,
        observer: Callable[[str, invocation_secrets.InvocationPublicationSnapshot], Awaitable[None]]
        | None,
        stager: DeferredTerminalStager | None,
    ) -> tuple[Event, runtime_records.ToolCallOutcome] | None:
        """Deliver an admission rejection without dispatching or running tool hooks."""

        if observer is not None:
            await observer(tool_call_id, snapshot)
        if stager is not None:
            await stager(event, outcome, False, False, snapshot)
            return None
        return await self.emit(event), outcome

    async def settle_admission_refusal(
        self,
        *,
        effect: ToolEffectRecord,
        session_store: SessionStore,
        run_epoch: int,
        execution_profile: ExecutionProfileIdentity | None,
        tool_round_identity: ToolRoundIdentity,
    ) -> None:
        """Publish the pre-dispatch refusal with its failed external-effect state."""

        # Selection, terminal evidence and the failed effect share one transaction.
        intent = effect.intent
        event = event_with_execution_profile_authority(
            Event(
                type=EventType.TOOL_CALL_FAILED,
                session_id=intent.session_id,
                interaction_id=intent.interaction_id,
                agent_name=intent.agent_name,
                environment_name=intent.environment_name,
                tool_name=intent.tool_name,
                payload={
                    **{
                        name: getattr(intent, name)
                        for name in (
                            "model_step_id",
                            "model_attempt_id",
                            "tool_round_id",
                            "tool_call_id",
                            "idempotency_key",
                        )
                    },
                    **(
                        {"approval_id": intent.approval_id}
                        if intent.approval_id is not None
                        else {}
                    ),
                    **({"input_id": intent.pause_id} if intent.pause_id is not None else {}),
                    "result": ToolResult(
                        content="Tool was not invoked because execution admission was refused.",
                        is_error=True,
                    ).model_dump(mode="json"),
                },
            ),
            execution_profile,
        )
        event = self._event_writer.prepare(
            _event_with_tool_round_authority(
                event,
                tool_round_identity,
                *(field for field in ("approval_id", "input_id") if field in event.payload),
            )
        )
        await ToolEffectStateOwner(session_store).transition(
            effect,
            state="failed",
            run_epoch=run_epoch,
            terminal=ToolEffectTerminal(
                event_id=event.id,
                result_digest=hashlib.sha256(
                    canonical_durable_json_bytes(event.payload["result"], "effect_terminal_result")
                ).hexdigest(),
            ),
            events=(event,),
        )
        await self._event_writer.fan_out_persisted([event])

    def detached_environment_work(self) -> set[asyncio.Future[Any]]:
        return set(self._detached_projections)

    def metrics(self) -> ToolTerminalPublicationMetricsSnapshot:
        """Return current content-free terminal-publication backlog and latency metrics."""

        return self.governor.snapshot()

    async def emit(self, event: Event, *, emitter: TerminalEventEmitter | None = None) -> Event:
        """Deliver one terminal through its selected persistence boundary."""

        if emitter is not None:
            return await emitter(event)
        return await self._event_writer.emit(event)

    async def emit_staged(self, event: Event) -> Event:
        """Prepare staged payload CPU off-loop, then use the writer's exact prepared path."""

        return await self._event_writer.emit_cooperatively(
            event,
            # Final preparation validates every event field. Conservatively
            # classify staged terminals as off-loop even when their content
            # string is small; nested arguments and metadata are not safely
            # measurable in constant time at this scheduling boundary.
            estimated_bytes=max(
                _terminal_publication_work_estimate(event),
                TOOL_TERMINAL_PUBLICATION_SLICE_BYTES + 1,
            ),
            scheduler=self.governor.run_cpu,
            persisted_observer=lambda persisted: self.governor.published(
                session_id=persisted.session_id,
                event_id=persisted.id,
                published_at=self._clock(),
            ),
        )

    async def emit_result(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        tool_call: runtime_records.ToolCallRequest,
        event_type: EventType,
        result: ToolResult,
        extra_payload: dict[str, Any],
        task_id: str | None,
        execution_profile: ExecutionProfileIdentity | None,
        invocation_context: InvocationContext | None,
        tool_round_identity: ToolRoundIdentity,
        approval_id: str | None,
        input_id: str | None,
        allow_modification: bool,
        redactor: SecretRedactor | None = None,
        output_redactor: SecretRedactor | None = None,
        argument_projection: tool_argument_publication.ToolArgumentProjection | None = None,
        deferred_terminal_stager: DeferredTerminalStager | None = None,
        publication_snapshot: invocation_secrets.InvocationPublicationSnapshot | None = None,
    ) -> AsyncIterator[tuple[Event, runtime_records.ToolCallOutcome | None]]:
        payload: dict[str, Any] = {
            "tool_call_id": tool_call.id,
            **extra_payload,
            "result": result.model_dump(),
            **copy_tool_round_identity(tool_round_identity).payload(),
        }
        if approval_id is not None:
            payload["approval_id"] = approval_id
        if input_id is not None:
            payload["input_id"] = input_id
        async for event in self.publish_result(
            event=Event(
                type=event_type,
                session_id=session.id,
                agent_name=registered_agent.spec.name,
                environment_name=_environment_name(registered_environment),
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
            redactor=redactor,
            output_redactor=output_redactor,
            argument_projection=argument_projection,
            allow_modification=allow_modification,
            deferred_terminal_stager=deferred_terminal_stager,
            publication_snapshot=publication_snapshot,
        ):
            yield event

    async def prepare_bounded_result(
        self,
        *,
        event: Event,
        result: ToolResult,
        registered_tool: runtime_records.RegisteredTool,
        session: Session,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        tool_call: runtime_records.ToolCallRequest,
        redactor: SecretRedactor,
    ) -> tuple[
        Event,
        ToolResult,
        tool_execution.ToolExecutionOutcome | None,
        asyncio.CancelledError | None,
        int,
    ]:
        """Admit a post-effect candidate only after projection or bounded failure.

        The candidate is internal and has not entered an event writer or store.
        Neither an oversized result nor its serializer error is durable evidence.
        """
        try:
            candidate = copy_event(event)
            if self._tool_result_projection_policy is None:
                validate_event_envelope(candidate)
            return candidate, result, None, None, 0
        except Exception as exc:
            limit_error = extract_durable_value_error(exc)
            if limit_error is None or limit_error.dimension is None:
                raise
        cancellation: asyncio.CancelledError | None = None
        if self._tool_result_projection_policy is not None:
            # Artifact projection sees only redacted output, and only the small
            # event scaffold crosses its await. The large body remains a result,
            # not an admitted Event payload.
            projected_input = tool_results.redact_tool_result(result, redactor)
            scaffold_payload = dict(event.payload)
            scaffold_payload.pop("result", None)
            scaffold = copy_event(event.model_copy(update={"payload": scaffold_payload}))
            return await self.project_result(
                event=scaffold,
                result=projected_input,
                session=session,
                registered_environment=registered_environment,
                tool_call=tool_call,
                effect=registered_tool.effect,
                redactor=redactor,
            )
        assert isinstance(limit_error, DurableValueError)
        failure = tool_execution.durable_output_limit_failure(
            error=limit_error,
            effect=registered_tool.effect,
            result=result,
            redactor=redactor,
        )
        failed = _bounded_tool_failure_event(event, failure)
        return failed, failure.result, failure, cancellation, 0

    async def _enforce_terminal_result_payload_limit(
        self,
        *,
        event: Event,
        result: ToolResult,
        registered_tool: runtime_records.RegisteredTool | None,
        redactor: SecretRedactor,
    ) -> tuple[Event, ToolResult, bool]:
        """Replace an oversized finalized result with bounded effect authority."""

        if registered_tool is None:
            return event, result, False
        maximum_bytes = ToolExecutionContract.model_validate(
            registered_tool.execution_contract
        ).max_terminal_payload_bytes
        if maximum_bytes is None:
            return event, result, False
        observed_bytes = await self.governor.run_cpu(
            _terminal_publication_work_estimate(event),
            lambda: _durable_payload_utf8_size(result.model_dump(mode="json")),
        )
        if observed_bytes <= maximum_bytes:
            return event, result, False

        failure = tool_execution.terminal_payload_limit_failure(
            effect=registered_tool.effect,
            maximum_bytes=maximum_bytes,
            observed_bytes=observed_bytes,
            redactor=redactor,
        )
        failure_result = failure.result
        failure_payload = dict(event.payload)
        for field_name in tool_terminal_controls.runtime_terminal_controls(failure_payload):
            failure_payload.pop(field_name, None)
        failure_payload.pop("tool_result_projection", None)
        failure_payload["result"] = failure_result.model_dump()
        failure_payload.update(failure.terminal_payload_fields())
        failed_event = event.model_copy(
            update={
                "type": EventType.TOOL_CALL_FAILED,
                "payload": failure_payload,
            }
        )
        failed_event, failure_result = _prepare_tool_result_event(
            event=failed_event,
            result=failure_result,
            redactor=redactor,
            runtime_tool=registered_tool.tool,
        )
        failure_bytes = _durable_payload_utf8_size(failure_result.model_dump(mode="json"))
        if failure_bytes > maximum_bytes:
            raise RuntimeError(
                "Registered terminal payload limit cannot contain its bounded "
                "contract-failure result."
            )
        return failed_event, failure_result, True

    @timed_phase("result_processing")
    async def publish_result(
        self,
        *,
        event: Event,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        tool_call: runtime_records.ToolCallRequest,
        result: ToolResult,
        task_id: str | None,
        redactor: SecretRedactor | None = None,
        output_redactor: SecretRedactor | None = None,
        argument_projection: tool_argument_publication.ToolArgumentProjection | None = None,
        hook_argument_projection: tool_argument_publication.ToolArgumentProjection | None = None,
        allow_modification: bool = False,
        publish_before_hooks: bool = False,
        deferred_terminal_stager: DeferredTerminalStager | None = None,
        deferred_terminal_projection_recorder: DeferredTerminalFinalizer | None = None,
        deferred_terminal_finalizer: DeferredTerminalFinalizer | None = None,
        terminal_event_emitter: TerminalEventEmitter | None = None,
        hooks_already_completed: bool = False,
        publication_snapshot: invocation_secrets.InvocationPublicationSnapshot | None = None,
        execution_profile: ExecutionProfileIdentity | None = None,
        invocation_context: InvocationContext | None = None,
        executed_runtime_tool: object | None = None,
    ) -> AsyncGenerator[tuple[Event, runtime_records.ToolCallOutcome | None], None]:
        if invocation_context is not None and (
            invocation_context.binding.session_id != session.id
            or invocation_context.registered_agent is not registered_agent
            or invocation_context.registered_environment is not registered_environment
            or invocation_context.profile is not execution_profile
        ):
            raise RuntimeError("Tool-result publication lost frozen invocation authority.")
        if publish_before_hooks and allow_modification:
            raise ValueError("Pre-hook tool-result publication cannot allow hook modification.")
        if hooks_already_completed and (publish_before_hooks or allow_modification):
            raise ValueError("Completed terminal hooks cannot be scheduled again.")
        if publish_before_hooks and "terminal_outcome" not in event.payload:
            # A runtime-owned isolated pre-dispatch rejection has boundary
            # controls, not a post-dispatch terminal outcome. Its evidence
            # must also precede observational hooks without becoming mutable.
            boundary_controls = tool_results.runtime_tool_execution_boundary_controls(event.payload)
            if "isolated_tool_failure_code" not in boundary_controls:
                raise ValueError("Pre-hook tool-result publication requires terminal controls.")
        if deferred_terminal_stager is not None and publication_snapshot is None:
            raise ValueError("Deferred terminal staging requires a publication snapshot.")
        if deferred_terminal_projection_recorder is not None and not publish_before_hooks:
            raise ValueError("Deferred terminal projection recording requires observational hooks.")

        registered_tool = registered_agent.executable_tool(tool_call.name)
        quarantine_hook_output = (
            registered_tool is not None and not registered_tool.publish_arguments
        )
        if quarantine_hook_output:
            allow_modification = False
        resolved_redactor = redactor if redactor is not None else self._secret_redactor
        resolved_output_redactor = resolved_redactor if output_redactor is None else output_redactor
        resolved_argument_projection = (
            tool_argument_publication.unavailable_argument_projection()
            if argument_projection is None
            else argument_projection
        )
        resolved_hook_argument_projection = (
            resolved_argument_projection
            if hook_argument_projection is None
            else hook_argument_projection
        )
        projected_tool_call = replace(
            tool_call,
            arguments=resolved_argument_projection.transcript_arguments(),
            arguments_state=resolved_argument_projection.state,
        )
        event = _restore_targeted_tool_invocation_event_authority(
            event,
            tool_call,
            redactor=resolved_redactor,
        )
        event_payload = dict(event.payload)
        event_payload.update(_published_argument_presence(tool_call, registered_tool))
        event_payload.pop(tool_argument_publication.ARGUMENTS_FIELD, None)
        event_payload.pop(tool_argument_publication.ARGUMENTS_STATE_FIELD, None)
        if resolved_argument_projection.state == "unavailable":
            event_payload.pop("effective_arguments", None)
            event_payload[tool_argument_publication.ARGUMENTS_EXACT_FIELD] = False
        event_payload.update(resolved_argument_projection.payload_fields())
        event = event.model_copy(update={"payload": event_payload})
        event = _event_with_targeted_tool_invocation_authority(event, tool_call)
        event = event_with_execution_profile_authority(event, execution_profile)
        hook_tool_call = _project_tool_call_for_hook(
            tool_call,
            argument_projection=resolved_hook_argument_projection,
            redactor=resolved_output_redactor,
        )
        event_identity_values = {
            field_name: event.payload.get(field_name)
            for field_name in ("model_step_id", "model_attempt_id", "tool_round_id")
        }
        if all(type(value) is str for value in event_identity_values.values()):
            event_identity = ToolRoundIdentity.model_validate(event_identity_values)
            event = _event_with_tool_round_authority(
                event,
                event_identity,
                *(
                    field_name
                    for field_name in (
                        "approval_id",
                        "input_id",
                        "profile_id",
                        "exposure_fingerprint",
                    )
                    if field_name in event.payload
                ),
            )
        boundary_controls, _boundary_references = tool_results.runtime_tool_event_boundary_controls(
            event.payload,
            include_terminal_controls=event.type
            in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED},
        )
        projection_record = boundary_controls.get("tool_result_projection")
        trusted_projection = (
            type(projection_record) is dict
            and type(projection_record.get("policy_id")) is str
            and event_nested_payload_authority_is_runtime_generated(
                event,
                path=_TOOL_RESULT_PROJECTION_PROVENANCE_PATH,
                value=projection_record["policy_id"],
            )
        )
        if trusted_projection:
            # Projection evidence and its strict artifact reference were
            # validated as one runtime-owned unit before durable staging.
            # Re-enter the schema-aware event boundary so a later secret scope
            # cannot rewrite the artifact identity during terminal replay.
            event = prepare_runtime_event(event, redactor=resolved_output_redactor)
            stored_result = event.payload.get("result")
            if type(stored_result) is not dict:
                raise RuntimeError("Projected terminal lost its tool result.")
            result = tool_results.tool_result_from_payload(stored_result)
            if not resolved_output_redactor.has_same_registry(resolved_redactor):
                event = prepare_runtime_event(event, redactor=resolved_redactor)
                stored_result = event.payload.get("result")
                if type(stored_result) is not dict:
                    raise RuntimeError("Projected terminal lost its tool result.")
                result = tool_results.tool_result_from_payload(stored_result)
        else:
            event, result = _prepare_tool_result_event(
                event=event,
                result=result,
                redactor=resolved_output_redactor,
                runtime_tool=executed_runtime_tool,
            )
            if not resolved_output_redactor.has_same_registry(resolved_redactor):
                event, result = _prepare_tool_result_event(
                    event=event,
                    result=result,
                    redactor=resolved_redactor,
                )
        event = _restore_targeted_tool_invocation_event_authority(
            event,
            tool_call,
            redactor=resolved_redactor,
        )
        pre_staging_projection_cancellation: asyncio.CancelledError | None = None
        pre_staging_projection_requests = 0
        projection_policy = getattr(self, "_tool_result_projection_policy", None)
        if (
            deferred_terminal_stager is not None
            and projection_policy is not None
            and not trusted_projection
        ):
            runtime_hooks = (
                self._hooks.registrations
                if invocation_context is None
                else invocation_context.runtime_hooks
            )
            has_after_tool_hooks = any(
                _runtime_hook_supports_phase(
                    hook=registered_hook.hook,
                    phase=RuntimeHookPhase.AFTER_TOOL_CALL,
                )
                for registered_hook in (*runtime_hooks, *registered_agent.runtime_hooks)
            )
            # Observational hooks already run after projection. With no
            # after-tool hooks there is likewise no modifier to preserve. In
            # both cases externalize before the durable stage so one large
            # completed result cannot inflate every later checkpoint copy.
            if publish_before_hooks or not has_after_tool_hooks:
                (
                    event,
                    result,
                    projection_size_failure,
                    pre_staging_projection_cancellation,
                    pre_staging_projection_requests,
                ) = await self.project_result(
                    event=event,
                    result=result,
                    session=session,
                    registered_environment=registered_environment,
                    tool_call=tool_call,
                    effect=ToolEffect.NONE if registered_tool is None else registered_tool.effect,
                    redactor=resolved_redactor,
                )
                if projection_size_failure is not None:
                    allow_modification = False
                    publish_before_hooks = True
                event = _restore_targeted_tool_invocation_event_authority(
                    event,
                    tool_call,
                    redactor=resolved_redactor,
                )
        event, result, result_limit_failed = await self._enforce_terminal_result_payload_limit(
            event=event,
            result=result,
            registered_tool=registered_tool,
            redactor=resolved_redactor,
        )
        if result_limit_failed:
            allow_modification = False
            publish_before_hooks = True
        event = _restore_targeted_tool_invocation_event_authority(
            event,
            tool_call,
            redactor=resolved_redactor,
        )
        if deferred_terminal_stager is not None:
            await deferred_terminal_stager(
                event,
                runtime_records.ToolCallOutcome(
                    call=projected_tool_call,
                    result=result,
                ),
                allow_modification,
                publish_before_hooks,
                cast(
                    "invocation_secrets.InvocationPublicationSnapshot",
                    publication_snapshot,
                ),
            )
            if pre_staging_projection_cancellation is not None:
                _raise_restored_post_tool_cancellation(
                    pre_staging_projection_cancellation,
                    restore_cancellation_requests=pre_staging_projection_requests,
                )
            return
        if hooks_already_completed:
            if deferred_terminal_finalizer is not None:
                # Recovery may conservatively suppress argument evidence even
                # when hooks already completed. Persist that final projection
                # before append, without invoking the hooks again.
                event = await deferred_terminal_finalizer(event)
                event = _restore_targeted_tool_invocation_event_authority(
                    event, tool_call, redactor=resolved_redactor
                )
                stored_result = event.payload.get("result")
                if type(stored_result) is not dict:
                    raise RuntimeError("Finalized staged terminal lost its tool result.")
                result = tool_results.tool_result_from_payload(stored_result)
            tool_event = await self.emit(event, emitter=terminal_event_emitter)
            yield (
                tool_event,
                runtime_records.ToolCallOutcome(
                    call=projected_tool_call,
                    result=result,
                ),
            )
            return
        if publish_before_hooks:
            if "tool_result_projection" in event.payload:
                projection_cancellation = None
                projection_requests = 0
            else:
                (
                    event,
                    result,
                    _projection_failure,
                    projection_cancellation,
                    projection_requests,
                ) = await self.project_result(
                    event=event,
                    result=result,
                    session=session,
                    registered_environment=registered_environment,
                    tool_call=hook_tool_call,
                    effect=ToolEffect.NONE if registered_tool is None else registered_tool.effect,
                    redactor=resolved_redactor,
                )
            event = _restore_targeted_tool_invocation_event_authority(
                event,
                tool_call,
                redactor=resolved_redactor,
            )
            if deferred_terminal_projection_recorder is not None:
                event = await deferred_terminal_projection_recorder(event)
                event = _restore_targeted_tool_invocation_event_authority(
                    event,
                    tool_call,
                    redactor=resolved_redactor,
                )
                stored_result = event.payload.get("result")
                if type(stored_result) is not dict:
                    raise RuntimeError("Projected staged terminal lost its tool result.")
                result = tool_results.tool_result_from_payload(stored_result)
            tool_event = await self.emit(event, emitter=terminal_event_emitter)
            yield (
                tool_event,
                runtime_records.ToolCallOutcome(
                    call=projected_tool_call,
                    result=result,
                ),
            )
            if projection_cancellation is not None:
                _raise_restored_post_tool_cancellation(
                    projection_cancellation,
                    restore_cancellation_requests=projection_requests,
                )
            async for hook_event, modified in self._hooks.after_call(
                session=session,
                tool_event=tool_event,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                tool_call=hook_tool_call,
                result=result,
                task_id=task_id,
                execution_profile=execution_profile,
                invocation_context=invocation_context,
                redactor=resolved_redactor,
                output_redactor=resolved_output_redactor,
                allow_modification=False,
                quarantine_output=quarantine_hook_output,
            ):
                if modified is not None:
                    raise AssertionError(
                        "Observational after-tool hook modified terminal evidence."
                    )
                yield hook_event, None
            if deferred_terminal_finalizer is not None:
                await deferred_terminal_finalizer(event)
            return
        final_result = result
        async for hook_event, modified in self._hooks.after_call(
            session=session,
            tool_event=event,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            tool_call=hook_tool_call,
            result=final_result,
            task_id=task_id,
            execution_profile=execution_profile,
            invocation_context=invocation_context,
            redactor=resolved_redactor,
            output_redactor=resolved_output_redactor,
            allow_modification=allow_modification,
            quarantine_output=quarantine_hook_output,
        ):
            yield hook_event, None
            if modified is not None:
                final_result = modified
        if final_result is not result:
            payload = dict(event.payload)
            payload["result"] = final_result.model_dump()
            event = event.model_copy(update={"payload": payload})
            event, final_result = _prepare_tool_result_event(
                event=event,
                result=final_result,
                redactor=resolved_output_redactor,
                restore_terminal_result_controls=False,
            )
            if not resolved_output_redactor.has_same_registry(resolved_redactor):
                event, final_result = _prepare_tool_result_event(
                    event=event,
                    result=final_result,
                    redactor=resolved_redactor,
                    restore_terminal_result_controls=False,
                )
        if "tool_result_projection" in event.payload and final_result is result:
            projection_cancellation = None
            projection_requests = 0
        else:
            (
                event,
                final_result,
                _projection_failure,
                projection_cancellation,
                projection_requests,
            ) = await self.project_result(
                event=event,
                result=final_result,
                session=session,
                registered_environment=registered_environment,
                tool_call=tool_call,
                effect=ToolEffect.NONE if registered_tool is None else registered_tool.effect,
                redactor=resolved_redactor,
            )
        (
            event,
            final_result,
            _result_limit_failed,
        ) = await self._enforce_terminal_result_payload_limit(
            event=event,
            result=final_result,
            registered_tool=registered_tool,
            redactor=resolved_redactor,
        )
        event = _restore_targeted_tool_invocation_event_authority(
            event,
            tool_call,
            redactor=resolved_redactor,
        )
        if deferred_terminal_finalizer is not None:
            event = await deferred_terminal_finalizer(event)
            event = _restore_targeted_tool_invocation_event_authority(
                event,
                tool_call,
                redactor=resolved_redactor,
            )
            stored_result = event.payload.get("result")
            if type(stored_result) is not dict:
                raise RuntimeError("Finalized staged terminal lost its tool result.")
            final_result = tool_results.tool_result_from_payload(stored_result)
        tool_event = await self.emit(event, emitter=terminal_event_emitter)
        yield (
            tool_event,
            runtime_records.ToolCallOutcome(
                call=projected_tool_call,
                result=final_result,
            ),
        )
        if projection_cancellation is not None:
            _raise_restored_post_tool_cancellation(
                projection_cancellation,
                restore_cancellation_requests=projection_requests,
            )

    def _retain_detached_projection(self, task: asyncio.Task[Any]) -> None:
        self._detached_projections.add(task)

        def settled(completed: asyncio.Task[Any]) -> None:
            self._detached_projections.discard(completed)
            _consume_projection_task_outcome(completed)

        task.add_done_callback(settled)

    async def project_result(
        self,
        *,
        event: Event,
        result: ToolResult,
        session: Session,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        tool_call: runtime_records.ToolCallRequest,
        effect: ToolEffect,
        redactor: SecretRedactor,
    ) -> tuple[
        Event,
        ToolResult,
        tool_execution.ToolExecutionOutcome | None,
        asyncio.CancelledError | None,
        int,
    ]:
        policy = self._tool_result_projection_policy
        if policy is None:
            try:
                validate_event_envelope(event)
            except Exception as exc:
                limit_error = extract_durable_value_error(exc)
                if limit_error is None or limit_error.dimension is None:
                    raise
                failure = tool_execution.durable_output_limit_failure(
                    error=limit_error,
                    effect=effect,
                    result=result,
                    redactor=redactor,
                )
                return _bounded_tool_failure_event(event, failure), failure.result, failure, None, 0
            return event, result, None, None, 0
        if event.payload.get("terminal_outcome") == "invalid_tool_output" and event.payload.get(
            "durable_value_error_code"
        ) in {"json_value_too_large", "too_many_json_nodes", "nesting_too_deep"}:
            # This is the bounded runtime disposition, not another candidate
            # for externalization. Re-running a policy here could create a
            # second artifact after the original projection already settled.
            return event, result, None, None, 0
        request = ToolResultProjectionRequest(
            result=result,
            session_id=session.id,
            agent_name=session.agent_name,
            environment_name=_environment_name(registered_environment),
            tool_call_id=tool_call.id,
            artifact_store=_artifact_store(registered_environment),
        )
        settlement_observer = ArtifactWriteSettlementObserver()
        with settlement_observer:
            policy_task = asyncio.create_task(policy.project(request))
        outcome = await await_shielded_task_outcome(
            policy_task,
            timeout_s=_TOOL_RESULT_PROJECTION_TIMEOUT_SECONDS,
        )
        error = outcome.error
        if outcome.timed_out:
            settlement_observer.record_active_candidates()
            policy_task.cancel()
            self._retain_detached_projection(policy_task)
            projection = projection_failure(
                policy=policy,
                request=request,
                failure_type="projection_timeout",
                artifact_write_settlement=_last_reconciliation_candidate(
                    settlement_observer.snapshot()
                ),
            )
        elif error is not None:
            settlement = _last_reconciliation_candidate(settlement_observer.snapshot())
            if isinstance(error, asyncio.CancelledError):
                error = unexpected_child_cancellation_error(
                    error,
                    operation="Tool-result projection policy",
                )
            if not isinstance(error, Exception):
                raise error
            projection = projection_failure(
                policy=policy,
                request=request,
                failure_type=safe_projection_failure_type(
                    error,
                    fallback="projection_policy_failure",
                ),
                artifact_write_settlement=settlement,
            )
        elif outcome.result is None:
            projection = projection_failure(
                policy=policy,
                request=request,
                failure_type="missing_projection_result",
                artifact_write_settlement=_last_reconciliation_candidate(
                    settlement_observer.snapshot()
                ),
            )
        else:
            try:
                projection = validate_tool_result_projection(
                    outcome.result,
                    request=request,
                    policy=policy,
                    observed_artifact_write_settlements=settlement_observer.snapshot(),
                )
            except Exception as exc:
                projection = projection_failure(
                    policy=policy,
                    request=request,
                    failure_type=safe_projection_failure_type(
                        exc,
                        fallback="projection_policy_failure",
                    ),
                    artifact_write_settlement=_last_reconciliation_candidate(
                        settlement_observer.snapshot()
                    ),
                )
        payload = dict(event.payload)
        payload["result"] = projection.result.model_dump()
        payload["tool_result_projection"] = projection.record.model_dump(
            mode="json",
            exclude_none=True,
        )
        projected_event = event.model_copy(update={"payload": payload})
        # The policy receives the final redacted result. Reapplying generic
        # redaction here would rewrite runtime-owned artifact identities when
        # a registered secret happens to overlap an id, hash, type, or status.
        try:
            projected_event, projected_result = _validate_and_synchronize_tool_result_event(
                event=projected_event,
                result=projection.result,
            )
            validate_event_envelope(projected_event)
        except Exception as exc:
            limit_error = extract_durable_value_error(exc)
            if limit_error is None or limit_error.dimension is None:
                raise
            failure = tool_execution.durable_output_limit_failure(
                error=limit_error,
                effect=effect,
                result=projection.result,
                redactor=redactor,
                projection_evidence=projection.record.model_dump(mode="json", exclude_none=True),
            )
            failed_event = _bounded_tool_failure_event(event, failure)
            return (
                failed_event,
                failure.result,
                failure,
                outcome.cancellation,
                outcome.cancellation_requests_consumed,
            )
        projected_event, projected_result = web_access_results.restore_attested_tool_result(
            projected_event,
            original=projected_result,
            redacted=projected_result,
        )
        projected_event, projected_result = shared_artifact_results.restore_attested_tool_result(
            projected_event,
            original=projected_result,
            redacted=projected_result,
        )
        projected_event = event_with_runtime_nested_payload_authority(
            projected_event,
            _TOOL_RESULT_PROJECTION_PROVENANCE_PATH,
        )
        return (
            projected_event,
            projected_result,
            None,
            outcome.cancellation,
            outcome.cancellation_requests_consumed,
        )
