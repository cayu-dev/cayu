"""Own ordinary tool-round staging, capacity and exact terminal publication.

Execution supplies tool events and outcomes. This owner retains the publication
evidence and enforces the ordering between private stages, public terminals and
the atomic transcript/checkpoint commit. Policy decisions and session-level
interruption closure remain with their existing owners.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Mapping
from contextlib import aclosing
from dataclasses import replace
from datetime import datetime
from typing import Any, Protocol

from cayu.events import Event, EventType, copy_event
from cayu.messages import Message
from cayu.runtime import _invocation_secrets as invocation_secrets
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _tool_argument_publication as tool_argument_publication
from cayu.runtime import _tool_results as tool_results
from cayu.runtime import _tool_round_publication as tool_round_publication
from cayu.runtime import _tool_round_recovery as tool_round_recovery
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime._invocation_lifecycle import InvocationContext
from cayu.runtime._tool_effect_state import ToolEffectReconciliationRequired, ToolEffectStateOwner
from cayu.runtime._tool_round_staging import (
    _staged_terminal_argument_projections,
    _terminal_publication_work_estimate,
    _ToolRoundPublicationCoordinator,
)
from cayu.runtime.execution_profiles import ExecutionProfileIdentity
from cayu.runtime.execution_units import ToolRoundIdentity, copy_tool_round_identity
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


class DurableToolRound:
    """One ordinary round's capacity lease and durable publication evidence."""

    def __init__(
        self,
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
    ) -> None:
        self._session = session
        self._identity = copy_tool_round_identity(tool_round_identity)
        self._tool_calls = tool_calls
        self.outcomes = outcomes
        self._session_store = session_store
        self._event_writer = event_writer
        self._registered_agent = registered_agent
        self._registered_environment = registered_environment
        self._task_id = task_id
        self._execution_profile = execution_profile
        self._invocation_context = invocation_context
        self._governor = publication_governor
        self._clock = clock
        self._emit_result = emit_result
        self._emit_terminal = emit_terminal
        self._admitted = False
        self._durable_events: list[Event] = []
        self._staged_hook_modes: dict[str, tuple[bool, bool]] = {}
        self._staged_private_outcomes: dict[str, runtime_records.ToolCallOutcome] = {}
        self._coordinator = (
            _ToolRoundPublicationCoordinator(
                session_id=session.id,
                session_instance_id=session.instance_id,
                run_epoch=session.run_epoch,
                tool_round_identity=self._identity,
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

    @property
    def defers_terminals(self) -> bool:
        return self._coordinator is not None

    async def admit(self) -> None:
        """Reserve the complete private round before the caller dispatches tools."""

        if self._admitted:
            raise RuntimeError("Tool round is already admitted.")
        if self._coordinator is not None:
            await self._coordinator.reserve_capacity()
        self._admitted = True

    def finish_dispatch(self) -> None:
        """Keep uncertain or durable stages fenced when dispatch stops."""

        if self._coordinator is not None:
            self._coordinator.seal_capacity()

    def _require_admitted(self) -> None:
        if not self._admitted:
            raise RuntimeError("Tool-round publication requires admission.")

    def observe_execution(
        self, event: Event, outcome: runtime_records.ToolCallOutcome | None = None
    ) -> None:
        """Retain only the current round's evidence after the caller yields it."""

        self._require_admitted()
        if event.type == EventType.TOOL_CALL_STARTED or (
            event.type in tool_round_recovery._TOOL_ROUND_TERMINAL_EVENT_TYPES
        ):
            self._durable_events.append(copy_event(event))
        if outcome is not None:
            self.outcomes.append(outcome)

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

        self._require_admitted()
        self.finish_dispatch()
        if self._coordinator is not None:
            async with aclosing(
                self._publish_staged_terminals({call.id for call in self._tool_calls})
            ) as terminals:
                async for event in terminals:
                    yield event
        source_checkpoint, pending_round = await self._load_pending_round(
            "The durable pending tool round changed before publication."
        )
        prepared = tool_round_publication.prepare_tool_round_publication(
            session_id=self._session.id,
            pending_round=pending_round,
            source_checkpoint=source_checkpoint,
            durable_events=self._durable_events,
            expected_statuses={SessionStatus.RUNNING, SessionStatus.INTERRUPTING},
            expected_run_epoch=self._session.run_epoch,
            expected_transcript_cursor=await self._session_store.load_transcript_cursor(
                self._session.id
            ),
        )
        cancellation = await tool_round_publication.publish_tool_round_with_exact_replay(
            prepared, session_store=self._session_store, event_writer=self._event_writer
        )
        messages.extend(prepared.request.transcript_messages)
        if cancellation is not None:
            raise cancellation

    async def publish_completed_effects(self) -> AsyncIterator[Event]:
        """Publish the known stages when workspace settlement stops execution."""

        self.finish_dispatch()
        if self._coordinator is not None:
            async for event in self._publish_staged_terminals(set(self._staged_private_outcomes)):
                yield event

    async def _synchronize_staged_outcomes(self) -> None:
        """Refresh private limit/interruption bookkeeping from durable stages."""

        if self._coordinator is None:
            return
        _, pending = await self._load_pending_round(
            "Staged outcomes lost their pending tool-round owner."
        )
        calls_by_id = {call.id: call for call in self._tool_calls}
        self._staged_private_outcomes.clear()
        for staged in pending.staged_terminals:
            call = calls_by_id.get(staged.tool_call_id)
            result_payload = staged.event.payload.get("result")
            if call is None or type(result_payload) is not dict:
                raise RuntimeError("Staged outcome conflicts with its tool-round call.")
            self._staged_private_outcomes[staged.tool_call_id] = runtime_records.ToolCallOutcome(
                call=replace(call, arguments={}, arguments_state="unavailable"),
                result=tool_results.tool_result_from_payload(result_payload),
            )
        self.outcomes[:] = [
            self._staged_private_outcomes[call.id]
            for call in self._tool_calls
            if call.id in self._staged_private_outcomes
        ]

    async def record_publication_snapshot(
        self,
        tool_call_id: str,
        snapshot: invocation_secrets.InvocationPublicationSnapshot,
    ) -> None:
        self._require_admitted()
        if self._coordinator is not None:
            await self._coordinator.seal_call(
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
        self._require_admitted()
        if self._coordinator is None:
            raise AssertionError("Round redactor observer requires a publication coordinator.")
        await self._coordinator.register_redactor(
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
        self._require_admitted()
        if self._coordinator is None:
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
        staged_event = await self._coordinator.stage_terminal(
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
        self._staged_hook_modes[outcome.call.id] = (
            (False if interrupted_terminal else allow_modification),
            (False if interrupted_terminal else publish_before_hooks),
        )
        await self._synchronize_staged_outcomes()
        return staged_event

    async def _complete_terminal_hooks(self, event: Event) -> Event:
        if self._coordinator is None:
            raise AssertionError("Hook finalization requires a publication coordinator.")
        return await self._coordinator.complete_terminal_hooks(event)

    async def _record_terminal_projection(self, event: Event) -> Event:
        if self._coordinator is None:
            raise AssertionError("Projection recording requires a publication coordinator.")
        return await self._coordinator.record_projected_terminal(event)

    async def record_workspace_capture(self, event: Event) -> Event:
        self._require_admitted()
        if self._coordinator is None:
            raise AssertionError("Workspace capture recording requires a publication coordinator.")
        recorded = await self._coordinator.record_workspace_capture(event)
        await self._synchronize_staged_outcomes()
        return recorded

    async def _publish_staged_terminals(
        self,
        expected_stage_ids: set[str],
    ) -> AsyncGenerator[Event, None]:
        coordinator = self._coordinator
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
                and expected_stage_ids.issubset({call.id for call in self._tool_calls})
                and await ToolEffectStateOwner(self._session_store).preserve_unresolved(
                    self._session,
                    tool_round_id=self._identity.tool_round_id,
                    tool_call_ids=tuple(call.id for call in self._tool_calls if call.id in missing),
                )
            ):
                raise ToolEffectReconciliationRequired()
            raise RuntimeError(
                "Dynamic multi-call publication has an unexpected staged-terminal set."
            )
        calls_by_id = {call.id: call for call in self._tool_calls}
        if not expected_stage_ids.issubset(calls_by_id):
            raise RuntimeError("Staged terminal publication names an unknown tool call.")
        final_outcomes: dict[str, runtime_records.ToolCallOutcome] = {}
        for tool_call in self._tool_calls:
            staged = staged_by_id.get(tool_call.id)
            if staged is None:
                continue
            staged = await coordinator.start_publication(staged)
            staged_bytes = staged.payload_bytes or _terminal_publication_work_estimate(staged.event)
            staged_event = await self._governor.run_cpu(
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
            registered_tool = self._registered_agent.executable_tool(tool_call.name)
            if (
                len(self._tool_calls) > 1
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
                else self._staged_hook_modes.get(
                    tool_call.id,
                    (
                        staged.hooks_state == "pending",
                        staged.hooks_state == "observational",
                    ),
                )
            )
            terminal_stream = self._emit_result(
                event=staged_event,
                session=self._session,
                registered_agent=self._registered_agent,
                registered_environment=self._registered_environment,
                tool_call=tool_call,
                result=result,
                task_id=self._task_id,
                execution_profile=self._execution_profile,
                invocation_context=self._invocation_context,
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
                terminal_event_emitter=self._emit_terminal,
                hooks_already_completed=hooks_already_completed,
            )
            async with aclosing(terminal_stream) as terminal_events:
                async for event, outcome in terminal_events:
                    if event.type in tool_round_recovery._TOOL_ROUND_TERMINAL_EVENT_TYPES:
                        self._governor.published(
                            session_id=self._session.id,
                            event_id=event.id,
                            published_at=self._clock(),
                        )
                        coordinator.terminal_published(event.id)
                    yield event
                    if event.type in tool_round_recovery._TOOL_ROUND_TERMINAL_EVENT_TYPES:
                        self._durable_events.append(copy_event(event))
                    if outcome is not None:
                        final_outcomes[outcome.call.id] = outcome
        if set(final_outcomes) != expected_stage_ids:
            raise RuntimeError("Staged terminal publication lost a public outcome.")
        current_by_id = {outcome.call.id: outcome for outcome in self.outcomes}
        current_by_id.update(final_outcomes)
        self.outcomes[:] = [
            current_by_id[call.id] for call in self._tool_calls if call.id in current_by_id
        ]

    async def publish_before_limit(self) -> AsyncIterator[Event]:
        if self._coordinator is not None:
            self.finish_dispatch()
        expected_stage_ids = {outcome.call.id for outcome in self.outcomes}
        async for event in self._publish_staged_terminals(expected_stage_ids):
            yield event

    async def publish_before_interrupt(self) -> AsyncIterator[Event]:
        """Make already completed effects authoritative before round interruption."""

        if self._coordinator is None:
            return
        self.finish_dispatch()
        if not self._staged_private_outcomes:
            return
        async for event in self._publish_staged_terminals(set(self._staged_private_outcomes)):
            yield event
