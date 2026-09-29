"""Own live tool-round staging and exact ordinary or limit publication.

Execution supplies tool events and outcomes. This owner retains the publication
evidence and enforces the ordering between private stages, public terminals and
the atomic transcript/checkpoint commit. Policy decisions and session-level
interruption closure remain with their existing owners.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Mapping
from contextlib import aclosing
from dataclasses import dataclass, field, replace
from datetime import datetime
from decimal import Decimal
from typing import Any, Protocol

from cayu._validation import MAX_DURABLE_JSON_INTEGER, MIN_DURABLE_JSON_INTEGER
from cayu.events import Event, EventType, copy_event
from cayu.messages import Message
from cayu.runtime import _invocation_secrets as invocation_secrets
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _tool_argument_publication as tool_argument_publication
from cayu.runtime import _tool_execution as tool_execution
from cayu.runtime import _tool_results as tool_results
from cayu.runtime import _tool_round_publication as tool_round_publication
from cayu.runtime import _tool_round_recovery as tool_round_recovery
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime._invocation_lifecycle import InvocationContext
from cayu.runtime._tool_effect_state import ToolEffectReconciliationRequired, ToolEffectStateOwner
from cayu.runtime._tool_round_staging import (
    _redactor_for_tool_calls,
    _staged_terminal_argument_projections,
    _terminal_publication_work_estimate,
    _ToolRoundPublicationCoordinator,
)
from cayu.runtime.execution_profiles import (
    ExecutionProfileIdentity,
    event_with_execution_profile_authority,
)
from cayu.runtime.execution_units import ToolRoundIdentity, copy_tool_round_identity
from cayu.runtime.stop_policy import StopDecision
from cayu.sessions.base import Session, SessionStatus, SessionStore
from cayu.tools._redaction import InvocationRedactorSnapshot
from cayu.tools.base import ToolResult
from cayu.tools.exposure import ResolvedToolExposureAuthority
from cayu.tools.terminal_publication import ToolTerminalPublicationGovernor
from cayu.vaults.redaction import SecretRedactor


class ToolTerminalPublisher(Protocol):
    """Execute existing result hooks using the round owner's durable callbacks."""

    def __call__(
        self,
        *,
        event: Event,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        tool_call: runtime_records.ToolCallRequest,
        result: ToolResult,
        task_id: str | None,
        execution_profile: ExecutionProfileIdentity | None,
        invocation_context: InvocationContext | None,
        redactor: SecretRedactor,
        output_redactor: SecretRedactor,
        argument_projection: tool_argument_publication.ToolArgumentProjection,
        hook_argument_projection: tool_argument_publication.ToolArgumentProjection,
        allow_modification: bool,
        publish_before_hooks: bool,
        deferred_terminal_projection_recorder: Callable[[Event], Awaitable[Event]] | None,
        deferred_terminal_finalizer: Callable[[Event], Awaitable[Event]] | None,
        terminal_event_emitter: Callable[[Event], Awaitable[Event]],
        hooks_already_completed: bool,
    ) -> AsyncGenerator[tuple[Event, runtime_records.ToolCallOutcome | None], None]: ...


@dataclass(frozen=True)
class DeferredInputMaterialization:
    messages: list[Message]
    cancellation: asyncio.CancelledError | None


class DeferredInputMaterializer(Protocol):
    def __call__(
        self,
        session_id: str,
        expected_messages: list[Message],
        *,
        cancellation: asyncio.CancelledError | None = None,
    ) -> Awaitable[DeferredInputMaterialization]: ...


@dataclass(slots=True)
class _ToolRoundExecution:
    """State needed only while dispatching and staging a live ordinary round."""

    tool_calls: list[runtime_records.ToolCallRequest]
    outcomes: list[runtime_records.ToolCallOutcome]
    registered_agent: runtime_records.RegisteredAgentState
    registered_environment: runtime_records.RegisteredEnvironment | None
    task_id: str | None
    execution_profile: ExecutionProfileIdentity | None
    invocation_context: InvocationContext | None
    publication_governor: ToolTerminalPublicationGovernor
    clock: Callable[[], datetime]
    emit_result: ToolTerminalPublisher
    emit_terminal: Callable[[Event], Awaitable[Event]]
    coordinator: _ToolRoundPublicationCoordinator | None
    admitted: bool = False
    durable_events: list[Event] = field(default_factory=list)
    staged_hook_modes: dict[str, tuple[bool, bool]] = field(default_factory=dict)
    staged_private_outcomes: dict[str, runtime_records.ToolCallOutcome] = field(
        default_factory=dict
    )


class DurableToolRound:
    """One durable round's publication, with staging state for ordinary execution."""

    def __init__(
        self,
        *,
        session: Session,
        tool_round_identity: ToolRoundIdentity,
        session_store: SessionStore,
        event_writer: RuntimeEventWriter,
    ) -> None:
        self._session = session
        self._identity = copy_tool_round_identity(tool_round_identity)
        self._session_store = session_store
        self._event_writer = event_writer
        self._execution: _ToolRoundExecution | None = None

    @classmethod
    def for_execution(
        cls,
        *,
        session: Session,
        tool_round_identity: ToolRoundIdentity,
        tool_calls: list[runtime_records.ToolCallRequest],
        outcomes: list[runtime_records.ToolCallOutcome],
        session_store: SessionStore,
        event_writer: RuntimeEventWriter,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        task_id: str | None,
        execution_profile: ExecutionProfileIdentity | None,
        invocation_context: InvocationContext | None,
        redactor: SecretRedactor,
        tool_exposure: ResolvedToolExposureAuthority | None,
        publication_governor: ToolTerminalPublicationGovernor,
        clock: Callable[[], datetime],
        emit_result: ToolTerminalPublisher,
        emit_terminal: Callable[[Event], Awaitable[Event]],
        defer_terminals: bool,
        terminal_payload_limits: Mapping[str, int | None] | None,
    ) -> DurableToolRound:
        """Attach the dispatch and staging dependencies for an ordinary round."""

        owner = cls(
            session=session,
            tool_round_identity=tool_round_identity,
            session_store=session_store,
            event_writer=event_writer,
        )
        coordinator = (
            _ToolRoundPublicationCoordinator(
                session_id=session.id,
                session_instance_id=session.instance_id,
                run_epoch=session.run_epoch,
                tool_round_identity=owner._identity,
                session_store=session_store,
                redactor=redactor,
                execution_profile=execution_profile,
                tool_exposure=tool_exposure,
                publication_governor=publication_governor,
                clock=clock,
                terminal_payload_limits=terminal_payload_limits,
            )
            if defer_terminals
            else None
        )
        owner._execution = _ToolRoundExecution(
            tool_calls=tool_calls,
            outcomes=outcomes,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            task_id=task_id,
            execution_profile=execution_profile,
            invocation_context=invocation_context,
            publication_governor=publication_governor,
            clock=clock,
            emit_result=emit_result,
            emit_terminal=emit_terminal,
            coordinator=coordinator,
        )
        return owner

    @property
    def defers_terminals(self) -> bool:
        return self._execution is not None and self._execution.coordinator is not None

    @property
    def outcomes(self) -> list[runtime_records.ToolCallOutcome]:
        return self._require_execution().outcomes

    def _require_execution(self) -> _ToolRoundExecution:
        if self._execution is None:
            raise RuntimeError("Tool round has no execution admission.")
        return self._execution

    async def admit(self) -> None:
        """Reserve the complete private round before the caller dispatches tools."""
        execution = self._require_execution()

        if execution.admitted:
            raise RuntimeError("Tool round is already admitted.")
        if execution.coordinator is not None:
            await execution.coordinator.reserve_capacity()
        execution.admitted = True

    def finish_dispatch(self) -> None:
        """Keep uncertain or durable stages fenced when dispatch stops."""
        execution = self._require_execution()

        if execution.coordinator is not None:
            execution.coordinator.seal_capacity()

    def _require_admitted(self) -> _ToolRoundExecution:
        execution = self._execution
        if execution is None or not execution.admitted:
            raise RuntimeError("Tool-round publication requires admission.")
        return execution

    def observe_execution(
        self, event: Event, outcome: runtime_records.ToolCallOutcome | None = None
    ) -> None:
        """Retain only the current round's evidence after the caller yields it."""

        execution = self._require_admitted()
        if event.type == EventType.TOOL_CALL_STARTED or (
            event.type in tool_round_recovery._TOOL_ROUND_TERMINAL_EVENT_TYPES
        ):
            execution.durable_events.append(copy_event(event))
        if outcome is not None:
            execution.outcomes.append(outcome)

    async def _load_pending_round(
        self, failure: str
    ) -> tuple[dict[str, Any] | None, tool_round_recovery.PendingToolRound]:
        """Read one fresh snapshot; publication uses that same validated input."""

        checkpoint = await self._session_store.load_checkpoint(self._session.id)
        pending = tool_round_recovery.pending_tool_round_from_checkpoint(checkpoint)
        if (
            pending is None
            or tool_round_recovery.pending_tool_round_identity(pending) != self._identity
        ):
            raise RuntimeError(failure)
        return checkpoint, pending

    async def publish(self, messages: list[Message]) -> AsyncGenerator[Event, None]:
        """Seal, publish ordered terminals, then commit the exact durable round."""

        execution = self._require_admitted()
        self.finish_dispatch()
        if execution.coordinator is not None:
            async with aclosing(
                self._publish_staged_terminals({call.id for call in execution.tool_calls})
            ) as terminals:
                async for event in terminals:
                    yield event
        source_checkpoint, pending_round = await self._load_pending_round(
            "The durable pending tool round changed before publication."
        )
        prepared, cancellation = await self._commit_snapshot(
            source_checkpoint, pending_round, execution.durable_events
        )
        messages.extend(prepared.request.transcript_messages)
        if cancellation is not None:
            raise cancellation

    async def _commit_snapshot(
        self,
        source_checkpoint: dict[str, Any] | None,
        pending_round: tool_round_recovery.PendingToolRound,
        durable_events: list[Event],
    ) -> tuple[tool_round_publication.PreparedToolRoundPublication, asyncio.CancelledError | None]:
        """Commit the supplied fresh snapshot and retain its exact replay request."""

        prepared = tool_round_publication.prepare_tool_round_publication(
            session_id=self._session.id,
            pending_round=pending_round,
            source_checkpoint=source_checkpoint,
            durable_events=durable_events,
            expected_statuses={SessionStatus.RUNNING, SessionStatus.INTERRUPTING},
            expected_run_epoch=self._session.run_epoch,
            expected_transcript_cursor=await self._session_store.load_transcript_cursor(
                self._session.id
            ),
        )
        cancellation = await tool_round_publication.publish_tool_round_with_exact_replay(
            prepared, session_store=self._session_store, event_writer=self._event_writer
        )
        return prepared, cancellation

    async def close_for_limit(
        self,
        *,
        messages: list[Message],
        tool_calls: list[runtime_records.ToolCallRequest],
        completed_tool_outcomes: list[runtime_records.ToolCallOutcome],
        decision: StopDecision,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        execution_profile: ExecutionProfileIdentity | None,
        redactor: SecretRedactor,
        materialize_deferred_input_if_present: Callable[[str], Awaitable[bool]],
        materialize_expected_deferred_input: DeferredInputMaterializer,
        raise_if_interrupted: Callable[[str], Awaitable[None]],
    ) -> AsyncGenerator[Event, None]:
        """Publish skipped calls and close a limited round once, retaining completed effects."""

        session = self._session
        tool_round_identity = self._identity
        tool_round_id = tool_round_identity.tool_round_id
        publication_id = f"tool-round:{tool_round_id}"
        if (
            await self._session_store.load_runtime_publication_receipt(
                session.id,
                publication_id,
            )
            is not None
        ):
            await materialize_deferred_input_if_present(session.id)
            messages[:] = await self._session_store.load_transcript(session.id)
            return

        completed_ids = {outcome.call.id for outcome in completed_tool_outcomes}
        remaining_tool_calls = [
            tool_call for tool_call in tool_calls if tool_call.id not in completed_ids
        ]
        skipped_outcomes = _limit_reached_tool_round_results(
            tool_calls=remaining_tool_calls,
            decision=decision,
            tool_round_identity=tool_round_identity,
        )
        completed_tool_outcomes = tool_results.redact_tool_call_outcomes(
            completed_tool_outcomes,
            redactor,
        )
        skipped_outcomes = tool_results.redact_tool_call_outcomes(
            skipped_outcomes,
            redactor,
        )
        base_round_redactor = _redactor_for_tool_calls(
            redactor,
            registered_agent=registered_agent,
            tool_calls=tool_calls,
        )
        for skipped_outcome in skipped_outcomes:
            await self._session_store.transform_checkpoint(
                session.id,
                lambda _current_session, current_checkpoint, call_id=skipped_outcome.call.id: (
                    tool_round_recovery.checkpoint_with_assistant_publication_snapshot(
                        current_checkpoint,
                        tool_round_identity=tool_round_identity,
                        tool_call_id=call_id,
                        redactor=base_round_redactor,
                        unsafe_output=False,
                    )
                ),
            )
            yield await self._event_writer.emit(
                _limit_reached_tool_call_event(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    tool_call_outcome=skipped_outcome,
                    decision=decision,
                    tool_round_identity=tool_round_identity,
                    execution_profile=execution_profile,
                )
            )

        source_checkpoint, pending_round = await self._load_pending_round(
            "Limited tool round lost its durable pending marker."
        )
        lifecycle_events = await self._session_store.load_tool_round_lifecycle_events_for_round(
            session.id,
            [call.tool_call_id for call in pending_round.tool_calls],
            tool_round_identity=tool_round_recovery.pending_tool_round_identity(pending_round),
        )
        _, cancellation = await self._commit_snapshot(
            source_checkpoint, pending_round, lifecycle_events
        )
        materialized = await materialize_expected_deferred_input(
            session.id,
            pending_round.deferred_messages,
            cancellation=cancellation,
        )
        messages[:] = materialized.messages
        cancellation = materialized.cancellation
        if cancellation is not None:
            raise cancellation
        await raise_if_interrupted(session.id)

    async def publish_completed_effects(self) -> AsyncIterator[Event]:
        """Publish the known stages when workspace settlement stops execution."""
        execution = self._require_execution()

        self.finish_dispatch()
        if execution.coordinator is not None:
            async for event in self._publish_staged_terminals(
                set(execution.staged_private_outcomes)
            ):
                yield event

    async def _synchronize_staged_outcomes(self) -> None:
        """Refresh private limit/interruption bookkeeping from durable stages."""
        execution = self._require_execution()

        if execution.coordinator is None:
            return
        _, pending = await self._load_pending_round(
            "Staged outcomes lost their pending tool-round owner."
        )
        calls_by_id = {call.id: call for call in execution.tool_calls}
        execution.staged_private_outcomes.clear()
        for staged in pending.staged_terminals:
            call = calls_by_id.get(staged.tool_call_id)
            result_payload = staged.event.payload.get("result")
            if call is None or type(result_payload) is not dict:
                raise RuntimeError("Staged outcome conflicts with its tool-round call.")
            execution.staged_private_outcomes[staged.tool_call_id] = (
                runtime_records.ToolCallOutcome(
                    call=replace(call, arguments={}, arguments_state="unavailable"),
                    result=tool_results.tool_result_from_payload(result_payload),
                )
            )
        execution.outcomes[:] = [
            execution.staged_private_outcomes[call.id]
            for call in execution.tool_calls
            if call.id in execution.staged_private_outcomes
        ]

    async def record_publication_snapshot(
        self,
        tool_call_id: str,
        snapshot: invocation_secrets.InvocationPublicationSnapshot,
    ) -> None:
        execution = self._require_admitted()
        if execution.coordinator is not None:
            await execution.coordinator.seal_call(
                tool_call_id=tool_call_id,
                snapshot=snapshot,
            )
            return
        await self._session_store.transform_checkpoint(
            self._session.id,
            tool_round_recovery.assistant_publication_snapshot_transform(
                tool_round_identity=self._identity,
                tool_call_id=tool_call_id,
                redactor=snapshot.redactor,
                unsafe_output=snapshot.secret_scope_incomplete,
            ),
        )

    async def record_redactor(
        self,
        tool_call_id: str,
        snapshot: InvocationRedactorSnapshot,
    ) -> None:
        execution = self._require_admitted()
        if execution.coordinator is None:
            raise AssertionError("Round redactor observer requires a publication coordinator.")
        await execution.coordinator.register_redactor(
            tool_call_id=tool_call_id,
            redactor=snapshot.redactor,
        )
        await self._synchronize_staged_outcomes()

    async def stage_terminal(
        self,
        event: Event,
        outcome: runtime_records.ToolCallOutcome,
        allow_modification: bool,
        publish_before_hooks: bool,
        snapshot: invocation_secrets.InvocationPublicationSnapshot,
    ) -> Event:
        execution = self._require_admitted()
        if execution.coordinator is None:
            raise AssertionError("Terminal staging requires a publication coordinator.")
        prepared_event = self._event_writer.prepare_candidate(event)
        interrupted_terminal = prepared_event.payload.get("interrupted") is True
        exposure_blocked = (
            prepared_event.type is EventType.TOOL_CALL_BLOCKED
            and prepared_event.payload.get("blocked_by") == "tool_exposure"
        )
        pre_execution_authority_rejected = (
            prepared_event.type is EventType.TOOL_CALL_FAILED
            and prepared_event.payload.get("blocked_by")
            in {"mcp_catalogue_authority", "targeted_tool_gateway", "targeted_tool_native"}
        )
        staged_event = await execution.coordinator.stage_terminal(
            tool_call_id=outcome.call.id,
            event=prepared_event,
            snapshot=snapshot,
            hooks_state=(
                "completed"
                if exposure_blocked or pre_execution_authority_rejected
                else (
                    "pending"
                    if interrupted_terminal
                    else (
                        "observational"
                        if publish_before_hooks
                        else ("pending" if allow_modification else "finalized")
                    )
                )
            ),
        )
        execution.staged_hook_modes[outcome.call.id] = (
            (False if interrupted_terminal else allow_modification),
            (False if interrupted_terminal else publish_before_hooks),
        )
        await self._synchronize_staged_outcomes()
        return staged_event

    async def _complete_terminal_hooks(self, event: Event) -> Event:
        execution = self._require_execution()
        if execution.coordinator is None:
            raise AssertionError("Hook finalization requires a publication coordinator.")
        return await execution.coordinator.complete_terminal_hooks(event)

    async def _record_terminal_projection(self, event: Event) -> Event:
        execution = self._require_execution()
        if execution.coordinator is None:
            raise AssertionError("Projection recording requires a publication coordinator.")
        return await execution.coordinator.record_projected_terminal(event)

    async def record_workspace_capture(self, event: Event) -> Event:
        execution = self._require_admitted()
        if execution.coordinator is None:
            raise AssertionError("Workspace capture recording requires a publication coordinator.")
        recorded = await execution.coordinator.record_workspace_capture(event)
        await self._synchronize_staged_outcomes()
        return recorded

    async def _publish_staged_terminals(
        self,
        expected_stage_ids: set[str],
    ) -> AsyncGenerator[Event, None]:
        execution = self._require_execution()
        coordinator = execution.coordinator
        if coordinator is None:
            if expected_stage_ids:
                raise AssertionError("Staged publication requires a coordinator.")
            return
        _, staged_round = await self._load_pending_round(
            "Staged terminal publication lost its pending tool round."
        )
        staged_by_id = {item.tool_call_id: item for item in staged_round.staged_terminals}
        if set(staged_by_id) != expected_stage_ids:
            missing = expected_stage_ids - set(staged_by_id)
            if (
                set(staged_by_id).issubset(expected_stage_ids)
                and expected_stage_ids.issubset({call.id for call in execution.tool_calls})
                and await ToolEffectStateOwner(self._session_store).preserve_unresolved(
                    self._session,
                    tool_round_id=self._identity.tool_round_id,
                    tool_call_ids=tuple(
                        call.id for call in execution.tool_calls if call.id in missing
                    ),
                )
            ):
                raise ToolEffectReconciliationRequired()
            raise RuntimeError(
                "Dynamic multi-call publication has an unexpected staged-terminal set."
            )
        calls_by_id = {call.id: call for call in execution.tool_calls}
        if not expected_stage_ids.issubset(calls_by_id):
            raise RuntimeError("Staged terminal publication names an unknown tool call.")
        final_outcomes: dict[str, runtime_records.ToolCallOutcome] = {}
        for tool_call in execution.tool_calls:
            staged = staged_by_id.get(tool_call.id)
            if staged is None:
                continue
            staged = await coordinator.start_publication(staged)
            staged_bytes = staged.payload_bytes or _terminal_publication_work_estimate(staged.event)
            staged_event = await execution.publication_governor.run_cpu(
                staged_bytes,
                lambda staged=staged: coordinator.restore_started_publication_authority(staged),
            )
            result_payload = staged_event.payload.get("result")
            if type(result_payload) is not dict:
                raise RuntimeError("Staged terminal publication lost its tool result.")
            result = tool_results.tool_result_from_payload(result_payload)
            argument_projection, hook_argument_projection = _staged_terminal_argument_projections(
                staged_event
            )
            registered_tool = execution.registered_agent.executable_tool(tool_call.name)
            if (
                len(execution.tool_calls) > 1
                and registered_tool is not None
                and registered_tool.publish_arguments
                and staged_event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
            ):
                argument_projection = tool_argument_publication.finalized_argument_projection(
                    tool_call.arguments,
                    redactor=coordinator.redactor,
                    scope_finalized=coordinator.argument_scope_finalized,
                )
                staged_payload = dict(staged_event.payload)
                staged_payload[tool_argument_publication.ARGUMENTS_EXACT_FIELD] = (
                    tool_argument_publication.argument_projection_is_exact(
                        argument_projection,
                        private_arguments=tool_call.arguments,
                    )
                )
                staged_event = staged_event.model_copy(update={"payload": staged_payload})
            hooks_already_completed = staged.hooks_state == "completed"
            allow_modification, publish_before_hooks = (
                (False, False)
                if hooks_already_completed
                else execution.staged_hook_modes.get(
                    tool_call.id,
                    (
                        staged.hooks_state == "pending",
                        staged.hooks_state == "observational",
                    ),
                )
            )
            terminal_stream = execution.emit_result(
                event=staged_event,
                session=self._session,
                registered_agent=execution.registered_agent,
                registered_environment=execution.registered_environment,
                tool_call=tool_call,
                result=result,
                task_id=execution.task_id,
                execution_profile=execution.execution_profile,
                invocation_context=execution.invocation_context,
                redactor=coordinator.redactor,
                output_redactor=coordinator.redactor,
                argument_projection=argument_projection,
                hook_argument_projection=hook_argument_projection,
                allow_modification=allow_modification,
                publish_before_hooks=publish_before_hooks,
                deferred_terminal_projection_recorder=(
                    self._record_terminal_projection
                    if publish_before_hooks and not hooks_already_completed
                    else None
                ),
                deferred_terminal_finalizer=(
                    None if hooks_already_completed else self._complete_terminal_hooks
                ),
                terminal_event_emitter=execution.emit_terminal,
                hooks_already_completed=hooks_already_completed,
            )
            async with aclosing(terminal_stream) as terminal_events:
                async for event, outcome in terminal_events:
                    if event.type in tool_round_recovery._TOOL_ROUND_TERMINAL_EVENT_TYPES:
                        execution.publication_governor.published(
                            session_id=self._session.id,
                            event_id=event.id,
                            published_at=execution.clock(),
                        )
                        coordinator.terminal_published(event.id)
                    yield event
                    if event.type in tool_round_recovery._TOOL_ROUND_TERMINAL_EVENT_TYPES:
                        execution.durable_events.append(copy_event(event))
                    if outcome is not None:
                        final_outcomes[outcome.call.id] = outcome
        if set(final_outcomes) != expected_stage_ids:
            raise RuntimeError("Staged terminal publication lost a public outcome.")
        current_by_id = {outcome.call.id: outcome for outcome in execution.outcomes}
        current_by_id.update(final_outcomes)
        execution.outcomes[:] = [
            current_by_id[call.id] for call in execution.tool_calls if call.id in current_by_id
        ]

    async def publish_before_limit(self) -> AsyncIterator[Event]:
        execution = self._require_execution()
        if execution.coordinator is not None:
            self.finish_dispatch()
        expected_stage_ids = {outcome.call.id for outcome in execution.outcomes}
        async for event in self._publish_staged_terminals(expected_stage_ids):
            yield event

    async def publish_before_interrupt(self) -> AsyncIterator[Event]:
        """Make already completed effects authoritative before round interruption."""
        execution = self._require_execution()

        if execution.coordinator is None:
            return
        self.finish_dispatch()
        if not execution.staged_private_outcomes:
            return
        async for event in self._publish_staged_terminals(set(execution.staged_private_outcomes)):
            yield event


def _environment_name(
    registered_environment: runtime_records.RegisteredEnvironment | None,
) -> str | None:
    if registered_environment is None:
        return None
    return registered_environment.spec.name


def _limit_value_for_payload(value: int | Decimal) -> int | str:
    if type(value) is Decimal:
        return str(value)
    if type(value) is int:
        if MIN_DURABLE_JSON_INTEGER <= value <= MAX_DURABLE_JSON_INTEGER:
            return value
        return str(value)
    raise TypeError("limit payload value must be an int or Decimal.")


def _limit_reached_tool_round_results(
    *,
    tool_calls: list[runtime_records.ToolCallRequest],
    decision: StopDecision,
    tool_round_identity: ToolRoundIdentity,
) -> list[runtime_records.ToolCallOutcome]:
    identity = copy_tool_round_identity(tool_round_identity)
    outcomes: list[runtime_records.ToolCallOutcome] = []
    for tool_call in tool_calls:
        structured = {
            "skipped": True,
            "reason": "limit_reached",
            "limit": decision.limit.value,
            "maximum": _limit_value_for_payload(decision.maximum),
            "actual": _limit_value_for_payload(decision.actual),
            "tool_call_id": tool_call.id,
            "tool_name": tool_call.name,
            **identity.payload(),
        }
        outcomes.append(
            runtime_records.ToolCallOutcome(
                call=tool_call,
                result=ToolResult(
                    content="Tool call skipped because a run limit was reached.",
                    structured=structured,
                    is_error=True,
                ),
            )
        )
    return outcomes


def _limit_reached_tool_call_event(
    *,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    registered_environment: runtime_records.RegisteredEnvironment | None,
    tool_call_outcome: runtime_records.ToolCallOutcome,
    decision: StopDecision,
    tool_round_identity: ToolRoundIdentity,
    execution_profile: ExecutionProfileIdentity | None,
    approval_id: str | None = None,
) -> Event:
    identity = copy_tool_round_identity(tool_round_identity)
    payload = {
        "tool_call_id": tool_call_outcome.call.id,
        "idempotency_key": tool_execution.tool_idempotency_key(
            session_id=session.id,
            tool_round_id=identity.tool_round_id,
            tool_call_id=tool_call_outcome.call.id,
            approval_id=approval_id,
        ),
        "reason": "limit_reached",
        "limit": decision.limit.value,
        "result": tool_call_outcome.result.model_dump(),
        **tool_argument_publication.unavailable_argument_projection().payload_fields(),
        **identity.payload(),
    }
    if approval_id is not None:
        payload["approval_id"] = approval_id
    return event_with_execution_profile_authority(
        Event(
            type=EventType.TOOL_CALL_FAILED,
            session_id=session.id,
            agent_name=registered_agent.spec.name,
            environment_name=_environment_name(registered_environment),
            tool_name=tool_call_outcome.call.name,
            payload=payload,
        ),
        execution_profile,
    )
