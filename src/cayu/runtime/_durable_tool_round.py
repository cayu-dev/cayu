"""Own live staging and exact ordinary, structured, limit and recovered publication.

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
from typing import Any, Literal, Protocol, TypeVar

from cayu._validation import MAX_DURABLE_JSON_INTEGER, MIN_DURABLE_JSON_INTEGER
from cayu.approvals.tools import PendingToolCallApproval, ToolPolicyEvidence
from cayu.context.structured_output import StructuredOutputSpec, StructuredOutputValidation
from cayu.events import Event, EventType, copy_event
from cayu.execution_profiles import (
    ExecutionProfileIdentity,
    event_with_execution_profile_authority,
)
from cayu.messages import Message
from cayu.runtime import _approval_support as approval_support
from cayu.runtime import _invocation_secrets as invocation_secrets
from cayu.runtime import _resume_ledger as resume_ledger
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _structured_output_tool_round as structured_output_tool_round
from cayu.runtime import _tool_execution as tool_execution
from cayu.runtime import _tool_results as tool_results
from cayu.runtime import _tool_round_publication as tool_round_publication
from cayu.runtime import _tool_round_recovery as tool_round_recovery
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime._invocation_lifecycle import InvocationContext
from cayu.runtime._phase_timing import (
    current_builder,
    finish_owned_timing,
    phase_scope,
    seal_owned_dispatch,
    timed_owned_round,
    timed_owned_stage,
    timed_phase,
    timed_stream,
    timing_scope,
)
from cayu.runtime._tool_effect_state import (
    ToolEffectReconciliationRequired,
    ToolEffectRecord,
    ToolEffectStateOwner,
)
from cayu.runtime._tool_round_continuation import ToolRoundContinuation
from cayu.runtime._tool_round_staging import (
    ToolTerminalPublisher,
    _redactor_for_tool_calls,
    _staged_terminal_argument_projections,
    _terminal_publication_work_estimate,
    _tool_terminal_payload_limits,
    _ToolRoundPublicationCoordinator,
)
from cayu.runtime.execution_units import ToolRoundIdentity, copy_tool_round_identity
from cayu.runtime.stop_policy import StopDecision
from cayu.sessions import _pending_tool_round as pending_rounds
from cayu.sessions import _pending_tool_round_reader as pending_round_reader
from cayu.sessions.base import Session, SessionStatus, SessionStore
from cayu.tools import _argument_publication as tool_argument_publication
from cayu.tools._redaction import InvocationRedactorSnapshot
from cayu.tools.base import ToolEffect, ToolResult
from cayu.tools.catalogue import ToolExecutionContract
from cayu.tools.exposure import (
    NOT_EXPOSED_IN_REQUEST_REASON,
    ResolvedToolExposureAuthority,
    unexposed_tool_result,
)
from cayu.tools.terminal_publication import ToolTerminalPublicationGovernor
from cayu.vaults.redaction import SecretRedactor

_T = TypeVar("_T")


class NativeToolTerminalPublisher(Protocol):
    def __call__(
        self, *, session: Session, record: ToolEffectRecord, event: Event
    ) -> Awaitable[Event]: ...


class ToolCallRecoveryResolver(Protocol):
    def __call__(
        self,
        pending_tool_call: PendingToolCallApproval,
        tool_call: runtime_records.ToolCallRequest,
        /,
    ) -> Awaitable[tuple[ToolResult | None, ToolEffectRecord | None]]: ...


@dataclass(frozen=True)
class InterruptedToolRoundRequest:
    session: Session
    registered_agent: runtime_records.RegisteredAgentState
    registered_environment: runtime_records.RegisteredEnvironment | None
    messages: list[Message]
    tool_calls: list[runtime_records.ToolCallRequest]
    tool_outcomes: list[runtime_records.ToolCallOutcome]
    tool_round_identity: ToolRoundIdentity
    cancellation_artifacts: list[dict[str, Any]] | None
    cancellation_artifacts_by_id: dict[str, list[dict[str, Any]]] | None
    cancellation_redactors_by_id: dict[str, SecretRedactor] | None = None
    execution_profile: ExecutionProfileIdentity | None = None
    invocation_context: InvocationContext | None = None


@dataclass(frozen=True)
class InterruptedToolRoundSnapshot:
    checkpoint: dict[str, Any] | None
    pending_round: pending_rounds.PendingToolRound
    tool_calls: list[runtime_records.ToolCallRequest]
    expected_transcript_cursor: int


@dataclass(frozen=True)
class StructuredToolRoundPublication:
    """Committed validation and events; session control owns retained cancellation."""

    validation: StructuredOutputValidation
    events: tuple[Event, ...]
    cancellation: asyncio.CancelledError | None


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
    """One durable round's publication, with optional live execution staging state."""

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
        self._continuation: ToolRoundContinuation | None = None
        # Timing for a round whose stages are not inside a live ToolRoundRun.
        self._timing: Any = None

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

    @classmethod
    def for_continuation(
        cls,
        *,
        session: Session,
        tool_round_identity: ToolRoundIdentity,
        session_store: SessionStore,
        event_writer: RuntimeEventWriter,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_name: str | None,
        tool_calls: list[runtime_records.ToolCallRequest],
        task_id: str | None,
        execution_profile: ExecutionProfileIdentity,
        invocation_context: InvocationContext | None,
        redactor: SecretRedactor,
        tool_exposure: ResolvedToolExposureAuthority | None,
        publication_governor: ToolTerminalPublicationGovernor,
        clock: Callable[[], datetime],
        emit_result: ToolTerminalPublisher,
        emit_terminal: Callable[[Event], Awaitable[Event]],
        defer_terminals: bool,
        terminal_payload_limits: Mapping[str, int | None] | None,
        pause_authority: dict[str, str],
        idempotency_options: dict[str, str],
    ) -> DurableToolRound:
        """Attach the paused-round phase without changing ordinary staging semantics."""
        owner = cls(
            session=session,
            tool_round_identity=tool_round_identity,
            session_store=session_store,
            event_writer=event_writer,
        )
        owner._continuation = ToolRoundContinuation(
            session=session,
            tool_round_identity=owner._identity,
            session_store=session_store,
            event_writer=event_writer,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            environment_name=environment_name,
            tool_calls=tool_calls,
            task_id=task_id,
            execution_profile=execution_profile,
            invocation_context=invocation_context,
            redactor=redactor,
            tool_exposure=tool_exposure,
            publication_governor=publication_governor,
            clock=clock,
            emit_result=emit_result,
            emit_terminal=emit_terminal,
            defer_terminals=defer_terminals,
            terminal_payload_limits=terminal_payload_limits,
            pause_authority=pause_authority,
            idempotency_options=idempotency_options,
        )
        return owner

    def _require_continuation(self) -> ToolRoundContinuation:
        if self._continuation is None:
            raise RuntimeError("Tool round has no continuation admission.")
        return self._continuation

    @property
    def continuation_redactor(self) -> SecretRedactor | None:
        return self._require_continuation().redactor

    @timed_owned_stage("staging")
    async def record_continuation_scope(
        self,
        tool_call_id: str,
        *,
        execution_scope_unknown: bool = False,
    ) -> None:
        await self._require_continuation().record_static_scope(
            tool_call_id,
            execution_scope_unknown=execution_scope_unknown,
        )

    @timed_owned_stage("staging")
    async def fence_restarted_continuation(
        self,
        *,
        recorded_ids: set[str],
        resume_undispatched_siblings: bool = False,
    ) -> set[str]:
        return await self._require_continuation().fence_restarted(
            recorded_ids=recorded_ids,
            resume_undispatched_siblings=resume_undispatched_siblings,
        )

    @timed_owned_round(recovered=False, finish=False)
    async def publish_continuation(
        self,
        *,
        already_published_ids: set[str],
        restarted_staged_ids: set[str],
    ) -> AsyncGenerator[tuple[Event, runtime_records.ToolCallOutcome | None], None]:
        async with aclosing(
            self._require_continuation().publish(
                already_published_ids=already_published_ids,
                restarted_staged_ids=restarted_staged_ids,
            )
        ) as published:
            async for item in published:
                yield item

    def timed_continuation_dispatch(
        self,
        stream: AsyncGenerator[tuple[Event, runtime_records.ToolCallOutcome | None], None],
    ) -> AsyncGenerator[tuple[Event, runtime_records.ToolCallOutcome | None], None]:
        """Attribute a paused round's dispatch to this round's timing record."""
        return stream if self._timing is None else timed_stream(self._timing, stream)

    async def commit_continuation_close(self, close: Awaitable[_T]) -> _T:
        """Measure the caller-owned durable close as this paused round's commit."""
        builder = self._timing
        if builder is None:
            return await close
        with timing_scope(builder), phase_scope("round_commit"):
            result = await close
        finish_owned_timing(self, committed=True)
        return result

    def finish_continuation_timing(self) -> None:
        """Record a paused round whose close was not committed as incomplete."""
        finish_owned_timing(self)

    @property
    def defers_terminals(self) -> bool:
        if self._continuation is not None:
            return self._continuation.defers_terminals
        return self._execution is not None and self._execution.coordinator is not None

    @property
    def outcomes(self) -> list[runtime_records.ToolCallOutcome]:
        return self._require_execution().outcomes

    def _require_execution(self) -> _ToolRoundExecution:
        if self._execution is None:
            raise RuntimeError("Tool round has no execution admission.")
        return self._execution

    @timed_owned_stage("admission")
    async def admit(self) -> None:
        """Reserve the complete private round before the caller dispatches tools."""
        if self._continuation is not None:
            await self._continuation.admit()
            return
        execution = self._require_execution()

        if execution.admitted:
            raise RuntimeError("Tool round is already admitted.")
        if execution.coordinator is not None:
            await execution.coordinator.reserve_capacity()
        execution.admitted = True

    def finish_dispatch(self) -> None:
        """Keep uncertain or durable stages fenced when dispatch stops."""
        seal_owned_dispatch(self)
        if self._continuation is not None:
            self._continuation.finish_dispatch()
            return
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
    ) -> tuple[dict[str, Any] | None, pending_rounds.PendingToolRound]:
        """Read one fresh snapshot; publication uses that same validated input."""

        checkpoint, pending = await pending_round_reader.load_pending_tool_round(
            self._session_store,
            self._session.id,
        )
        if pending is None or pending_rounds.pending_tool_round_identity(pending) != self._identity:
            raise RuntimeError(failure)
        builder = current_builder()
        if builder is not None:
            builder.register_calls(pending.tool_calls)
        return checkpoint, pending

    @timed_owned_round(recovered=False)
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

    @timed_phase("round_commit")
    async def _commit_snapshot(
        self,
        source_checkpoint: dict[str, Any] | None,
        pending_round: pending_rounds.PendingToolRound,
        durable_events: list[Event],
        *,
        expected_statuses: set[SessionStatus] | None = None,
        expected_transcript_cursor: int | None = None,
        extension: tool_round_publication.ToolRoundPublicationExtension | None = None,
    ) -> tuple[tool_round_publication.PreparedToolRoundPublication, asyncio.CancelledError | None]:
        """Commit the supplied fresh snapshot and retain its exact replay request."""

        prepared = tool_round_publication.prepare_tool_round_publication(
            session_id=self._session.id,
            pending_round=pending_round,
            source_checkpoint=source_checkpoint,
            durable_events=durable_events,
            expected_statuses=(
                {SessionStatus.RUNNING, SessionStatus.INTERRUPTING}
                if expected_statuses is None
                else expected_statuses
            ),
            expected_run_epoch=self._session.run_epoch,
            expected_transcript_cursor=(
                await self._session_store.load_transcript_cursor(self._session.id)
                if expected_transcript_cursor is None
                else expected_transcript_cursor
            ),
            extension=extension,
        )
        cancellation = await tool_round_publication.publish_tool_round_with_exact_replay(
            prepared, session_store=self._session_store, event_writer=self._event_writer
        )
        builder = current_builder()
        if builder is not None:
            builder.mark_committed()
        return prepared, cancellation

    @timed_owned_round(recovered=False)
    async def publish_structured(
        self,
        *,
        registered_agent: runtime_records.RegisteredAgentState,
        environment_name: str | None,
        messages: list[Message],
        tool_calls: list[runtime_records.ToolCallRequest],
        pending_round: pending_rounds.PendingToolRound,
        spec: StructuredOutputSpec,
        step: int,
        attempt: int,
        retry_allowed: bool,
        execution_profile: ExecutionProfileIdentity | None,
        redactor: SecretRedactor,
        tool_redactor: SecretRedactor,
    ) -> AsyncGenerator[Event | StructuredToolRoundPublication, None]:
        """Publish live validation; return cancellation for session-level closure."""
        session = self._session
        tool_round_identity = self._identity
        validation = structured_output_tool_round._validate_structured_output_tool_round(
            tool_calls=tool_calls,
            spec=spec,
        )
        durable_validation = pending_round.structured_output_validation
        if durable_validation is None:
            raise RuntimeError("Structured-output tool round lost its authoritative validation.")
        if durable_validation != structured_output_tool_round._redact_structured_output_validation(
            validation,
            redactor,
        ):
            raise RuntimeError(
                "Structured-output validation conflicts with its durable model-completion evidence."
            )
        structured_tool_outcomes = (
            structured_output_tool_round._structured_output_tool_round_outcomes(
                tool_calls=tool_calls,
                spec=spec,
                validation=validation,
            )
        )
        structured_tool_outcomes = tool_results.redact_tool_call_outcomes(
            structured_tool_outcomes,
            redactor,
        )
        structured_round_redactor = _redactor_for_tool_calls(
            tool_redactor,
            registered_agent=registered_agent,
            tool_calls=tool_calls,
        )
        for outcome in structured_tool_outcomes:
            await self._session_store.transform_checkpoint(
                session.id,
                tool_round_recovery.assistant_publication_snapshot_transform(
                    tool_round_identity=tool_round_identity,
                    tool_call_id=outcome.call.id,
                    redactor=structured_round_redactor,
                    unsafe_output=False,
                ),
            )
            terminal_event = await self._event_writer.emit(
                event_with_execution_profile_authority(
                    structured_output_tool_round._structured_output_tool_terminal_event(
                        session=session,
                        registered_agent=registered_agent,
                        environment_name=environment_name,
                        tool_round_identity=tool_round_identity,
                        outcome=outcome,
                    ),
                    execution_profile,
                )
            )
            yield terminal_event

        extension = self._structured_output_extension(
            registered_agent=registered_agent,
            environment_name=environment_name,
            spec=spec,
            validation=validation,
            step=step,
            attempt=attempt,
            retry_allowed=retry_allowed,
            execution_profile=execution_profile,
            redactor=redactor,
        )
        source_checkpoint, durable_pending_round = await self._load_pending_round(
            "Structured-output tool round marker changed before publication."
        )
        lifecycle_events = await self._session_store.load_tool_round_lifecycle_events_for_round(
            session.id,
            [call.tool_call_id for call in durable_pending_round.tool_calls],
            tool_round_identity=pending_rounds.pending_tool_round_identity(durable_pending_round),
        )
        prepared, cancellation = await self._commit_snapshot(
            source_checkpoint, durable_pending_round, lifecycle_events, extension=extension
        )
        messages.extend(prepared.request.transcript_messages)
        yield StructuredToolRoundPublication(validation, extension.events, cancellation)

    # The same rebuild closes a structured round interrupted before its live runner.
    @timed_owned_round(recovered=lambda kwargs: not kwargs.get("interrupted", False))
    async def recover_structured(
        self,
        *,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        messages: list[Message],
        pending_round: pending_rounds.PendingToolRound,
        retry_allowed: bool,
        expected_transcript_cursor: int,
        execution_profile: ExecutionProfileIdentity | None,
        redactor: SecretRedactor,
        tool_redactor: SecretRedactor,
        materialize_expected_deferred_input: DeferredInputMaterializer,
        invocation_context: InvocationContext | None = None,
        interrupted: bool = False,
    ) -> AsyncGenerator[Event, None]:
        """Rebuild one reserved finalizer round from its durable model output.

        ``interrupted`` only labels the timing record of an interrupted close.
        """

        del interrupted
        session = self._session
        if invocation_context is not None and (
            invocation_context.binding.session_id != session.id
            or registered_agent is not invocation_context.registered_agent
            or registered_environment is not invocation_context.registered_environment
            or execution_profile is not invocation_context.profile
        ):
            raise RuntimeError(
                "Structured-output recovery substituted frozen invocation authority."
            )

        spec = pending_round.structured_output
        step = pending_round.model_step
        attempt = pending_round.structured_output_attempt
        if spec is None or step is None or attempt is None:
            raise RuntimeError(
                "Structured-output recovery requires durable config, step, and attempt."
            )
        if attempt > spec.max_retries + 1:
            raise RuntimeError(
                "Structured-output recovery attempt exceeds the durable retry policy."
            )
        tool_calls = tool_round_recovery.pending_round_tool_calls(pending_round)
        validation = pending_round.structured_output_validation
        if validation is None:
            raise RuntimeError(
                "Structured-output recovery requires authoritative durable validation."
            )
        validation = validation.model_copy(deep=True)
        expected_outcomes = structured_output_tool_round._structured_output_tool_round_outcomes(
            tool_calls=tool_calls,
            spec=spec,
            validation=validation,
        )
        expected_outcomes = tool_results.redact_tool_call_outcomes(
            expected_outcomes,
            redactor,
        )
        tool_round_identity = pending_rounds.pending_tool_round_identity(pending_round)
        structured_round_redactor = _redactor_for_tool_calls(
            tool_redactor,
            registered_agent=registered_agent,
            tool_calls=tool_calls,
        )
        for expected_outcome in expected_outcomes:
            await self._session_store.transform_checkpoint(
                session.id,
                tool_round_recovery.assistant_publication_snapshot_transform(
                    tool_round_identity=tool_round_identity,
                    tool_call_id=expected_outcome.call.id,
                    redactor=structured_round_redactor,
                    unsafe_output=False,
                ),
            )
        source_checkpoint, pending_round = await self._load_pending_round(
            "Structured-output tool round changed while sealing its publication projection."
        )
        environment_name = _environment_name(registered_environment)
        lifecycle_events = await tool_round_recovery.load_tool_round_lifecycle_events(
            self._session_store,
            session_id=session.id,
            pending_round=pending_round,
        )
        recorded_outcomes, _started_ids = tool_round_recovery.recorded_tool_outcomes(
            events=lifecycle_events,
            pending_round=pending_round,
        )
        terminal_events_by_call = {
            event.payload["tool_call_id"]: event
            for event in lifecycle_events
            if event.type
            in {
                EventType.TOOL_CALL_COMPLETED,
                EventType.TOOL_CALL_FAILED,
                EventType.TOOL_CALL_BLOCKED,
                EventType.TOOL_CALL_APPROVAL_DENIED,
            }
        }
        planned_terminal_events: list[Event] = []
        for expected_outcome in expected_outcomes:
            expected_event = event_with_execution_profile_authority(
                structured_output_tool_round._structured_output_tool_terminal_event(
                    session=session,
                    registered_agent=registered_agent,
                    environment_name=environment_name,
                    tool_round_identity=pending_rounds.pending_tool_round_identity(pending_round),
                    outcome=expected_outcome,
                ),
                execution_profile,
            )
            if invocation_context is not None:
                expected_event = expected_event.model_copy(
                    update={"interaction_id": invocation_context.binding.interaction_id}
                )
            recorded_outcome = recorded_outcomes.get(expected_outcome.call.id)
            if recorded_outcome is None:
                planned_terminal_events.append(expected_event)
                continue
            recorded_event = terminal_events_by_call.get(expected_outcome.call.id)
            expected_payload = expected_event.payload
            legacy_expected_payload = dict(expected_payload)
            legacy_expected_payload.pop(tool_argument_publication.ARGUMENTS_STATE_FIELD, None)
            recorded_payload_matches = recorded_event is not None and (
                recorded_event.payload == expected_payload
                or (
                    tool_argument_publication.ARGUMENTS_STATE_FIELD not in recorded_event.payload
                    and recorded_event.payload == legacy_expected_payload
                )
            )
            expected_recorded_outcome = expected_outcome
            if recorded_event is not None:
                recorded_projection = tool_argument_publication.terminal_argument_projection(
                    recorded_event.payload,
                    legacy_arguments=expected_outcome.call.arguments,
                )
                expected_recorded_outcome = runtime_records.ToolCallOutcome(
                    call=runtime_records.copy_tool_call_request(
                        expected_outcome.call,
                        arguments=recorded_projection.transcript_arguments(),
                        arguments_state=recorded_projection.state,
                    ),
                    result=expected_outcome.result,
                )
            if (
                recorded_outcome != expected_recorded_outcome
                or recorded_event is None
                or recorded_event.id != expected_event.id
                or recorded_event.type != expected_event.type
                or recorded_event.session_id != expected_event.session_id
                or (
                    invocation_context is not None
                    and recorded_event.interaction_id != expected_event.interaction_id
                )
                or recorded_event.agent_name != expected_event.agent_name
                or recorded_event.environment_name != expected_event.environment_name
                or recorded_event.tool_name != expected_event.tool_name
                or not recorded_payload_matches
            ):
                raise RuntimeError(
                    "Durable structured-output terminal evidence conflicts with "
                    f"the pending call: {expected_outcome.call.id}"
                )

        tool_round_publication.collect_tool_round_publication_evidence(
            session_id=session.id,
            pending_round=pending_round,
            durable_events=[*lifecycle_events, *planned_terminal_events],
        )
        emitted_terminal_events: list[Event] = []
        for terminal_event in planned_terminal_events:
            emitted_terminal_events.append(await self._event_writer.emit(terminal_event))

        lifecycle_events = await tool_round_recovery.load_tool_round_lifecycle_events(
            self._session_store,
            session_id=session.id,
            pending_round=pending_round,
        )
        extension = self._structured_output_extension(
            registered_agent=registered_agent,
            environment_name=environment_name,
            spec=spec,
            validation=validation,
            step=step,
            attempt=attempt,
            retry_allowed=retry_allowed,
            execution_profile=execution_profile,
            redactor=redactor,
            invocation_context=invocation_context,
        )
        _, cancellation = await self._commit_snapshot(
            source_checkpoint,
            pending_round,
            lifecycle_events,
            expected_statuses={
                SessionStatus.RUNNING,
                SessionStatus.INTERRUPTING,
                SessionStatus.INTERRUPTED,
            },
            expected_transcript_cursor=expected_transcript_cursor,
            extension=extension,
        )
        materialized = await materialize_expected_deferred_input(
            session.id,
            pending_round.deferred_messages,
            cancellation=cancellation,
        )
        messages[:] = materialized.messages
        cancellation = materialized.cancellation
        for event in emitted_terminal_events:
            yield event
        for event in extension.events:
            yield copy_event(event)
        if cancellation is not None:
            raise cancellation

    def _structured_output_extension(
        self,
        *,
        registered_agent: runtime_records.RegisteredAgentState,
        environment_name: str | None,
        spec: StructuredOutputSpec,
        validation: StructuredOutputValidation,
        step: int,
        attempt: int,
        retry_allowed: bool,
        execution_profile: ExecutionProfileIdentity | None,
        redactor: SecretRedactor,
        invocation_context: InvocationContext | None = None,
    ) -> structured_output_tool_round._StructuredOutputToolRoundPublicationExtension:
        """Bind validation events and retry intent to the exact round publication."""
        session = self._session
        tool_round_identity = self._identity
        retry_scheduled = not validation.valid and retry_allowed and attempt <= spec.max_retries
        validating_event = structured_output_tool_round._structured_output_validating_event(
            session=session,
            registered_agent=registered_agent,
            environment_name=environment_name,
            spec=spec,
            step=step,
            attempt=attempt,
            tool_round_identity=tool_round_identity,
        )
        outcome_event = structured_output_tool_round._structured_output_event(
            event_type=(
                EventType.STRUCTURED_OUTPUT_VALIDATED
                if validation.valid
                else EventType.STRUCTURED_OUTPUT_FAILED
            ),
            session=session,
            registered_agent=registered_agent,
            environment_name=environment_name,
            spec=spec,
            validation=validation,
            step=step,
            attempt=attempt,
            redactor=redactor,
            tool_round_identity=tool_round_identity,
        )
        auxiliary_events = [validating_event, outcome_event]
        if retry_scheduled:
            auxiliary_events.append(
                structured_output_tool_round._structured_output_event(
                    event_type=EventType.STRUCTURED_OUTPUT_RETRY,
                    session=session,
                    registered_agent=registered_agent,
                    environment_name=environment_name,
                    spec=spec,
                    validation=validation,
                    step=step,
                    attempt=attempt,
                    redactor=redactor,
                    tool_round_identity=tool_round_identity,
                )
            )
        auxiliary_events = self._event_writer.prepare_many(
            [
                event_with_execution_profile_authority(
                    event
                    if invocation_context is None
                    else event.model_copy(
                        update={"interaction_id": invocation_context.binding.interaction_id}
                    ),
                    execution_profile,
                )
                for event in auxiliary_events
            ]
        )
        return structured_output_tool_round._StructuredOutputToolRoundPublicationExtension(
            intent={
                "schema_version": 1,
                "kind": "structured-output-validation",
                "step": step,
                "attempt": attempt,
                "valid": validation.valid,
                "retry_scheduled": retry_scheduled,
                "event_ids": [event.id for event in auxiliary_events],
            },
            events=tuple(auxiliary_events),
        )

    @timed_owned_round(recovered=False)
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
            tool_round_identity=pending_rounds.pending_tool_round_identity(pending_round),
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

    async def prepare_interruption(
        self,
        request: InterruptedToolRoundRequest,
        *,
        materialize_deferred_input_if_present: Callable[[str], Awaitable[bool]],
    ) -> InterruptedToolRoundSnapshot | None:
        """Retain the original transcript cursor and exact round before settlement."""
        if request.invocation_context is not None and (
            request.invocation_context.binding.session_id != request.session.id
            or request.registered_agent is not request.invocation_context.registered_agent
            or request.registered_environment
            is not request.invocation_context.registered_environment
            or request.execution_profile is not request.invocation_context.profile
        ):
            raise RuntimeError(
                "Interrupted tool-round recovery substituted frozen invocation authority."
            )
        tool_round_identity = copy_tool_round_identity(request.tool_round_identity)
        publication_id = f"tool-round:{tool_round_identity.tool_round_id}"
        if (
            await self._session_store.load_runtime_publication_receipt(
                request.session.id,
                publication_id,
            )
            is not None
        ):
            await materialize_deferred_input_if_present(request.session.id)
            request.messages[:] = await self._session_store.load_transcript(request.session.id)
            return None
        expected_transcript_cursor = await self._session_store.load_transcript_cursor(
            request.session.id
        )
        source_checkpoint, pending_round = await self._load_pending_round(
            "Interrupted tool round lost its durable pending marker."
        )
        pending_tool_calls = tool_round_recovery.pending_round_tool_calls(pending_round)
        if [(tool_call.id, tool_call.name) for tool_call in request.tool_calls] != [
            (tool_call.id, tool_call.name) for tool_call in pending_tool_calls
        ]:
            raise RuntimeError("Interrupted tool calls conflict with the durable pending round.")
        return InterruptedToolRoundSnapshot(
            source_checkpoint, pending_round, pending_tool_calls, expected_transcript_cursor
        )

    async def _complete_recovery_assistant_publication(
        self,
        *,
        registered_agent: runtime_records.RegisteredAgentState,
        redactor: SecretRedactor,
        pending_round: pending_rounds.PendingToolRound,
        execution_scope_unknown_ids: set[str] | frozenset[str] = frozenset(),
    ) -> tuple[dict[str, Any], pending_rounds.PendingToolRound]:
        """Finalize calls after every returned secret was durably projected."""

        session_id = self._session.id
        if pending_round.assistant_message_state == "published":
            checkpoint = await self._session_store.load_checkpoint(session_id)
            return checkpoint or {}, pending_round
        identity = pending_rounds.pending_tool_round_identity(pending_round)
        tool_calls = tool_round_recovery.pending_round_tool_calls(pending_round)
        base_redactor = _redactor_for_tool_calls(
            redactor,
            registered_agent=registered_agent,
            tool_calls=tool_calls,
        )
        publication = pending_round.assistant_publication
        covered_ids = set() if publication is None else set(publication.covered_tool_call_ids)
        # Recovery trusts only the capability recorded with the original model
        # completion. Current environment registration may differ after a
        # restart; missing legacy evidence is therefore treated as unknown.
        secret_resolution_scope = (
            "unknown" if publication is None else publication.secret_resolution_scope
        )
        expected_ids = {tool_call.id for tool_call in tool_calls}
        if not execution_scope_unknown_ids <= expected_ids:
            raise RuntimeError("Assistant recovery evidence names a call outside its tool round.")
        for tool_call in tool_calls:
            if tool_call.id in covered_ids:
                continue
            await self._session_store.transform_checkpoint(
                session_id,
                tool_round_recovery.assistant_publication_snapshot_transform(
                    tool_round_identity=identity,
                    tool_call_id=tool_call.id,
                    redactor=base_redactor,
                    unsafe_output=(
                        secret_resolution_scope != "static"
                        and tool_call.id in execution_scope_unknown_ids
                    ),
                ),
            )
        checkpoint, recovered_round = await self._load_pending_round(
            "Pending tool round changed while sealing its recovery publication projection."
        )
        return checkpoint or {}, recovered_round

    @timed_owned_stage("effect_state", recovered=True)
    async def recover_outcomes(
        self,
        *,
        registered_agent: runtime_records.RegisteredAgentState,
        pending_round: pending_rounds.PendingToolRound,
        recorded_outcomes: Mapping[str, runtime_records.ToolCallOutcome],
        effective_started_ids: set[str],
        reconcile_call: ToolCallRecoveryResolver,
        redactor: SecretRedactor,
    ) -> tuple[list[runtime_records.ToolCallOutcome], dict[str, ToolEffectRecord]]:
        """Select missing round results after specialized reconciliation proves them safe.

        Recorded and staged calls are never reconciled again. The resolver owns
        native operation, external-effect and child-session evidence; it must
        raise when a call cannot safely receive an unknown-outcome terminal.
        """
        if pending_rounds.pending_tool_round_identity(pending_round) != self._identity:
            raise RuntimeError("Recovered outcomes belong to a different tool round.")
        synthesized_outcomes: list[runtime_records.ToolCallOutcome] = []
        confirmed_native_effect_records: dict[str, ToolEffectRecord] = {}
        staged_call_ids = {
            staged.tool_call_id
            for staged in tool_round_recovery.staged_terminal_records(pending_round)
        }
        for pending_tool_call in pending_round.tool_calls:
            if (
                recorded_outcomes.get(pending_tool_call.tool_call_id) is not None
                or pending_tool_call.tool_call_id in staged_call_ids
            ):
                continue
            tool_call = approval_support.tool_call_request_from_pending(pending_tool_call)
            if (
                approval_support.effective_tool_policy_evidence(pending_tool_call)
                is ToolPolicyEvidence.UNEXPOSED
            ):
                exposure = pending_round.tool_exposure
                if exposure is None:
                    raise RuntimeError(
                        "Unexposed recovered tool call lost its frozen exposure snapshot."
                    )
                if (
                    tool_call.name not in registered_agent.executable_tool_names
                    or tool_call.name in exposure.tool_names
                ):
                    raise RuntimeError(
                        "Unexposed recovered tool call conflicts with its frozen exposure."
                    )
                synthesized_outcomes.append(
                    runtime_records.ToolCallOutcome(
                        call=runtime_records.copy_tool_call_request(
                            tool_call,
                            arguments={},
                            arguments_state="unavailable",
                        ),
                        result=unexposed_tool_result(),
                    )
                )
                continue
            with phase_scope("effect_state", call_id=pending_tool_call.tool_call_id):
                result, confirmed_effect_record = await reconcile_call(pending_tool_call, tool_call)
            if result is None:
                registered_tool = registered_agent.executable_tool(pending_tool_call.tool_name)
                result = tool_round_recovery.unknown_recovered_tool_result(
                    pending_tool_call=pending_tool_call,
                    pending_round=pending_round,
                    started=pending_tool_call.tool_call_id in effective_started_ids,
                    effect=None if registered_tool is None else registered_tool.effect,
                )
            if confirmed_effect_record is not None:
                confirmed_native_effect_records[pending_tool_call.tool_call_id] = (
                    confirmed_effect_record
                )
            synthesized_outcomes.append(
                runtime_records.ToolCallOutcome(call=tool_call, result=result)
            )
        return (
            tool_results.redact_tool_call_outcomes(synthesized_outcomes, redactor),
            confirmed_native_effect_records,
        )

    # The same publication closes a round interrupted before its live runner.
    @timed_owned_round(recovered=lambda kwargs: not kwargs.get("interrupted", False))
    async def publish_recovered(
        self,
        *,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        messages: list[Message],
        pending_round: pending_rounds.PendingToolRound,
        lifecycle_events: list[Event],
        synthesized_outcomes: list[runtime_records.ToolCallOutcome],
        effective_started_ids: set[str],
        expected_transcript_cursor: int,
        execution_profile: ExecutionProfileIdentity | None,
        invocation_context: InvocationContext | None,
        redactor: SecretRedactor,
        publication_governor: ToolTerminalPublicationGovernor,
        clock: Callable[[], datetime],
        runtime_hooks: tuple[runtime_records.RegisteredRuntimeHook, ...],
        emit_result: ToolTerminalPublisher,
        emit_terminal: Callable[[Event], Awaitable[Event]],
        emit_native_terminal: NativeToolTerminalPublisher,
        materialize_expected_deferred_input: DeferredInputMaterializer,
        interrupted: bool = False,
        confirmed_native_effect_records: Mapping[str, ToolEffectRecord] | None = None,
    ) -> AsyncGenerator[Event, None]:
        """Publish safe staged outcomes and synthesized results without replaying tools."""

        session = self._session
        environment_name = _environment_name(registered_environment)
        # Classify staged evidence against the durable coverage exactly as it
        # existed when recovery took ownership. Completing the assistant
        # projection below may conservatively add coverage for calls that never
        # produced terminal evidence; that must not retroactively authenticate
        # a sibling result staged under an incomplete dynamic-secret scope.
        recovery_staged_records = tool_round_recovery.staged_terminal_records(pending_round)
        if any(item.event.session_id != session.id for item in recovery_staged_records):
            raise RuntimeError("Staged recovery evidence belongs to a different session.")
        checkpoint, pending_round = await self._complete_recovery_assistant_publication(
            registered_agent=registered_agent,
            redactor=redactor,
            pending_round=pending_round,
            execution_scope_unknown_ids=effective_started_ids,
        )
        if pending_round.assistant_message_state == "quarantined":
            tool_round_recovery.ready_assistant_publication_message(pending_round)
        publication_scope = (
            "unknown"
            if pending_round.assistant_publication is None
            else pending_round.assistant_publication.secret_resolution_scope
        )
        if publication_scope != "static":
            quarantined_records: list[tool_round_recovery.StagedToolCallTerminal] = []
            for staged in recovery_staged_records:
                if staged.hooks_state == "completed":
                    quarantined_records.append(staged)
                    continue
                quarantined_event = tool_round_recovery.hook_scope_unavailable_recovery_event(
                    staged.event
                )
                await self._session_store.transform_checkpoint(
                    session.id,
                    tool_round_recovery.completed_staged_terminal_transform(
                        tool_round_identity=pending_rounds.pending_tool_round_identity(
                            pending_round
                        ),
                        event=quarantined_event,
                    ),
                )
                quarantined_records.append(
                    staged.model_copy(
                        update={
                            "event": quarantined_event,
                            "hooks_state": "completed",
                        },
                        deep=True,
                    )
                )
            recovery_staged_records = quarantined_records
        synthesized_by_id = {outcome.call.id: outcome for outcome in synthesized_outcomes}
        staged_records_by_id = {item.tool_call_id: item for item in recovery_staged_records}
        staged_events_by_id = {
            tool_call_id: item.event for tool_call_id, item in staged_records_by_id.items()
        }
        staged_hook_states_by_id = {
            item.tool_call_id: item.hooks_state for item in recovery_staged_records
        }
        durable_terminal_ids = {
            event.payload.get("tool_call_id")
            for event in lifecycle_events
            if event.type in tool_round_recovery._TOOL_ROUND_TERMINAL_EVENT_TYPES
        }
        pending_calls_by_id = {call.tool_call_id: call for call in pending_round.tool_calls}
        started_interaction_by_id = {
            tool_call_id: event.interaction_id
            for event in lifecycle_events
            if event.type is EventType.TOOL_CALL_STARTED
            and type(tool_call_id := event.payload.get("tool_call_id")) is str
        }
        planned_terminal_events: list[Event] = []
        planned_outcomes: list[runtime_records.ToolCallOutcome] = []
        planned_hook_states: list[
            Literal["pending", "finalized", "observational", "completed"]
        ] = []
        for pending_call in pending_round.tool_calls:
            if pending_call.tool_call_id in durable_terminal_ids:
                continue
            staged_event = staged_events_by_id.get(pending_call.tool_call_id)
            if staged_event is not None:
                planned_terminal_events.append(staged_event)
                planned_outcomes.append(
                    resume_ledger.tool_call_outcome_from_terminal_event(
                        event=staged_event,
                        pending_tool_call=pending_calls_by_id[pending_call.tool_call_id],
                    )
                )
                planned_hook_states.append(staged_hook_states_by_id[pending_call.tool_call_id])
                continue
            outcome = synthesized_by_id.get(pending_call.tool_call_id)
            if outcome is None:
                raise RuntimeError("Recovery lost terminal evidence for a pending tool call.")
            if interrupted:
                planned_terminal_events.append(
                    _interrupted_tool_call_event(
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        tool_call_outcome=outcome,
                        tool_round_identity=pending_rounds.pending_tool_round_identity(
                            pending_round
                        ),
                    )
                )
                planned_outcomes.append(outcome)
                planned_hook_states.append("finalized")
                continue
            policy_evidence = approval_support.effective_tool_policy_evidence(pending_call)
            is_unexposed = policy_evidence is ToolPolicyEvidence.UNEXPOSED
            event_type = (
                EventType.TOOL_CALL_BLOCKED
                if is_unexposed
                else (
                    EventType.TOOL_CALL_FAILED
                    if outcome.result.is_error
                    else EventType.TOOL_CALL_COMPLETED
                )
            )
            exposure = pending_round.tool_exposure if is_unexposed else None
            if is_unexposed and exposure is None:
                raise RuntimeError(
                    "Unexposed recovered terminal lost its frozen exposure snapshot."
                )
            planned_terminal_events.append(
                Event(
                    type=event_type,
                    session_id=session.id,
                    interaction_id=started_interaction_by_id.get(pending_call.tool_call_id),
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    tool_name=outcome.call.name,
                    payload={
                        **pending_rounds.pending_tool_round_identity(pending_round).payload(),
                        "tool_call_id": outcome.call.id,
                        "idempotency_key": tool_execution.tool_idempotency_key(
                            session_id=session.id,
                            tool_round_id=pending_round.tool_round_id,
                            tool_call_id=outcome.call.id,
                        ),
                        "recovered": True,
                        **(
                            {}
                            if exposure is None
                            else {
                                "blocked_by": "tool_exposure",
                                "reason": NOT_EXPOSED_IN_REQUEST_REASON,
                                "profile_id": exposure.profile_id,
                                "exposure_fingerprint": exposure.fingerprint,
                            }
                        ),
                        **(
                            tool_argument_publication.unavailable_argument_projection().payload_fields()
                            if is_unexposed
                            else {}
                        ),
                        "result": outcome.result.model_dump(),
                    },
                )
            )
            planned_outcomes.append(outcome)
            planned_hook_states.append("completed" if is_unexposed else "finalized")
        tool_round_publication.collect_tool_round_publication_evidence(
            session_id=session.id,
            pending_round=pending_round,
            durable_events=[*lifecycle_events, *planned_terminal_events],
        )

        emitted_events: list[Event] = []
        tool_round_identity = pending_rounds.pending_tool_round_identity(pending_round)
        recovery_publication_coordinator = _ToolRoundPublicationCoordinator(
            session_id=session.id,
            session_instance_id=session.instance_id,
            run_epoch=session.run_epoch,
            tool_round_identity=tool_round_identity,
            session_store=self._session_store,
            redactor=_redactor_for_tool_calls(
                redactor,
                registered_agent=registered_agent,
                tool_calls=[item.call for item in planned_outcomes],
            ),
            execution_profile=execution_profile,
            tool_exposure=pending_round.tool_exposure,
            publication_governor=publication_governor,
            clock=clock,
            terminal_payload_limits=await _tool_terminal_payload_limits(
                registered_agent,
                [
                    approval_support.tool_call_request_from_pending(pending_call)
                    for pending_call in pending_round.tool_calls
                ],
                publication_governor=publication_governor,
                runtime_hooks=(
                    runtime_hooks
                    if invocation_context is None
                    else invocation_context.runtime_hooks
                ),
            ),
        )
        await recovery_publication_coordinator.reserve_capacity()
        await recovery_publication_coordinator.restore_staged_capacity(recovery_staged_records)
        for staged in recovery_staged_records:
            if staged.tool_call_id not in durable_terminal_ids:
                continue
            publication_governor.published(
                session_id=session.id,
                event_id=staged.event.id,
                published_at=clock(),
            )
            recovery_publication_coordinator.terminal_published(staged.event.id)
        recovery_publication_coordinator.seal_capacity()

        async def complete_recovered_terminal_hooks(event: Event) -> Event:
            await self._session_store.transform_checkpoint(
                session.id,
                tool_round_recovery.completed_staged_terminal_transform(
                    tool_round_identity=tool_round_identity,
                    event=event,
                ),
            )
            return copy_event(event)

        async def record_recovered_terminal_projection(event: Event) -> Event:
            await self._session_store.transform_checkpoint(
                session.id,
                tool_round_recovery.projected_staged_terminal_transform(
                    tool_round_identity=tool_round_identity,
                    event=event,
                ),
            )
            return copy_event(event)

        async def emit_confirmed_native_terminal(event: Event) -> Event:
            if confirmed_native_effect_records is None:
                raise RuntimeError("Native terminal publication lost its evidence owner.")
            record = confirmed_native_effect_records[event.payload["tool_call_id"]]
            return await emit_native_terminal(session=session, record=record, event=event)

        for expected_outcome, terminal_event, hooks_state in zip(
            planned_outcomes,
            planned_terminal_events,
            planned_hook_states,
            strict=True,
        ):
            if expected_outcome.call.id in staged_events_by_id:
                staged_record = await recovery_publication_coordinator.start_publication(
                    staged_records_by_id[expected_outcome.call.id]
                )
                terminal_event = await publication_governor.run_cpu(
                    staged_record.payload_bytes or 0,
                    lambda staged_record=staged_record: (
                        recovery_publication_coordinator.restore_started_publication_authority(
                            staged_record
                        )
                    ),
                )
            argument_projection = tool_argument_publication.unavailable_argument_projection()
            hook_argument_projection = argument_projection
            if expected_outcome.call.id in staged_events_by_id and publication_scope == "static":
                # Only a static scope survives restart with argument authority.
                # Dynamic scopes remain unavailable after their redactor is lost.
                # The sealed stage already owns the public argument projection.
                # Replacing it on retry would conflict with that same durable stage.
                argument_projection, hook_argument_projection = (
                    _staged_terminal_argument_projections(terminal_event)
                )
            expected_public_outcome = runtime_records.ToolCallOutcome(
                call=runtime_records.copy_tool_call_request(
                    expected_outcome.call,
                    arguments=argument_projection.transcript_arguments(),
                    arguments_state=argument_projection.state,
                ),
                result=expected_outcome.result,
            )
            terminal_stream = emit_result(
                event=terminal_event,
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                tool_call=expected_outcome.call,
                result=expected_outcome.result,
                task_id=pending_round.task_id,
                execution_profile=execution_profile,
                argument_projection=argument_projection,
                hook_argument_projection=hook_argument_projection,
                allow_modification=hooks_state == "pending",
                publish_before_hooks=hooks_state == "observational",
                deferred_terminal_projection_recorder=(
                    record_recovered_terminal_projection
                    if hooks_state == "observational"
                    and expected_outcome.call.id in staged_events_by_id
                    else None
                ),
                deferred_terminal_finalizer=(
                    complete_recovered_terminal_hooks
                    if hooks_state in {"pending", "finalized", "observational", "completed"}
                    and expected_outcome.call.id in staged_events_by_id
                    else None
                ),
                terminal_event_emitter=(
                    emit_terminal
                    if expected_outcome.call.id in staged_events_by_id
                    else emit_confirmed_native_terminal
                    if confirmed_native_effect_records is not None
                    and expected_outcome.call.id in confirmed_native_effect_records
                    else None
                ),
                hooks_already_completed=hooks_state == "completed",
                invocation_context=invocation_context,
            )
            async with aclosing(terminal_stream) as terminal_events:
                async for event, emitted_outcome in terminal_events:
                    emitted_events.append(event)
                    if event.type in tool_round_recovery._TOOL_ROUND_TERMINAL_EVENT_TYPES:
                        publication_governor.published(
                            session_id=session.id,
                            event_id=event.id,
                            published_at=clock(),
                        )
                        recovery_publication_coordinator.terminal_published(event.id)
                    if (
                        emitted_outcome is not None
                        and hooks_state != "pending"
                        and emitted_outcome != expected_public_outcome
                    ):
                        raise RuntimeError("Recovered tool-round hooks changed terminal evidence.")

        lifecycle_events = await tool_round_recovery.load_tool_round_lifecycle_events(
            self._session_store,
            session_id=session.id,
            pending_round=pending_round,
        )
        checkpoint, pending_round = await self._load_pending_round(
            "Recovered hooks lost their pending tool-round owner."
        )
        _, cancellation = await self._commit_snapshot(
            checkpoint,
            pending_round,
            lifecycle_events,
            expected_statuses={
                SessionStatus.RUNNING,
                SessionStatus.INTERRUPTING,
                SessionStatus.INTERRUPTED,
            },
            expected_transcript_cursor=expected_transcript_cursor,
        )
        materialized = await materialize_expected_deferred_input(
            session.id,
            pending_round.deferred_messages,
            cancellation=cancellation,
        )
        messages[:] = materialized.messages
        cancellation = materialized.cancellation
        for event in emitted_events:
            yield event
        if cancellation is not None:
            raise cancellation

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

    @timed_owned_stage("staging")
    async def record_publication_snapshot(
        self,
        tool_call_id: str,
        snapshot: invocation_secrets.InvocationPublicationSnapshot,
    ) -> None:
        if self._continuation is not None:
            await self._continuation.record_publication_snapshot(tool_call_id, snapshot)
            return
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

    @timed_owned_stage("staging")
    async def record_redactor(
        self,
        tool_call_id: str,
        snapshot: InvocationRedactorSnapshot,
    ) -> None:
        if self._continuation is not None:
            await self._continuation.record_redactor(tool_call_id, snapshot)
            return
        execution = self._require_admitted()
        if execution.coordinator is None:
            raise AssertionError("Round redactor observer requires a publication coordinator.")
        await execution.coordinator.register_redactor(
            tool_call_id=tool_call_id,
            redactor=snapshot.redactor,
        )
        await self._synchronize_staged_outcomes()

    @timed_owned_stage("staging")
    async def stage_terminal(
        self,
        event: Event,
        outcome: runtime_records.ToolCallOutcome,
        allow_modification: bool,
        publish_before_hooks: bool,
        snapshot: invocation_secrets.InvocationPublicationSnapshot,
    ) -> Event:
        if self._continuation is not None:
            return await self._continuation.stage_terminal(
                event, outcome, allow_modification, publish_before_hooks, snapshot
            )
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

    @timed_owned_stage("staging")
    async def record_workspace_capture(self, event: Event) -> Event:
        if self._continuation is not None:
            return await self._continuation.record_workspace_capture(event)
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


def _interrupted_tool_call_outcome(
    *,
    tool_call: runtime_records.ToolCallRequest,
    tool_round_identity: ToolRoundIdentity,
    registered_tool: runtime_records.RegisteredTool | None,
    execution_started: bool,
    artifacts: list[dict[str, Any]] | None = None,
) -> runtime_records.ToolCallOutcome:
    """Build the canonical bounded result for one interrupted tool call."""

    structured = {
        "interrupted": True,
        "tool_call_id": tool_call.id,
        "tool_name": tool_call.name,
        **copy_tool_round_identity(tool_round_identity).payload(),
    }
    if execution_started and registered_tool is not None:
        try:
            execution_contract = ToolExecutionContract.model_validate(
                registered_tool.execution_contract
            )
        except (TypeError, ValueError):
            execution_contract = None
        if execution_contract is not None and execution_contract.boundary == "posix_process":
            process_controls = {
                "terminal_outcome": "tool_execution_error",
                "tool_effect": registered_tool.effect.value,
                "outcome_unknown": registered_tool.effect is not ToolEffect.NONE,
                "manual_reconciliation_required": (registered_tool.effect is ToolEffect.EXTERNAL),
                "isolated_tool_failure_code": "process_interrupted",
                "tool_execution_boundary": "posix_process",
                "tool_timeout_strength": "hard_process_deadline",
            }
            tool_results.runtime_terminal_controls(process_controls)
            tool_results.runtime_tool_execution_boundary_controls(process_controls)
            structured.update(process_controls)
    return runtime_records.ToolCallOutcome(
        call=tool_call,
        result=ToolResult(
            content="Tool call interrupted before completion.",
            structured=structured,
            artifacts=[] if artifacts is None else artifacts,
            is_error=True,
        ),
    )


def _interrupted_tool_call_event(
    *,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    registered_environment: runtime_records.RegisteredEnvironment | None,
    tool_call_outcome: runtime_records.ToolCallOutcome,
    tool_round_identity: ToolRoundIdentity,
) -> Event:
    """Build the terminal event paired with the canonical interrupted result."""

    structured = dict(tool_call_outcome.result.structured or {})
    terminal_controls = tool_results.runtime_terminal_controls(structured)
    terminal_controls.update(tool_results.runtime_tool_execution_boundary_controls(structured))
    return Event(
        type=(
            EventType.TOOL_CALL_FAILED
            if tool_call_outcome.result.is_error
            else EventType.TOOL_CALL_COMPLETED
        ),
        session_id=session.id,
        agent_name=registered_agent.spec.name,
        environment_name=_environment_name(registered_environment),
        tool_name=tool_call_outcome.call.name,
        payload={
            "tool_call_id": tool_call_outcome.call.id,
            "idempotency_key": tool_execution.tool_idempotency_key(
                session_id=session.id,
                tool_round_id=tool_round_identity.tool_round_id,
                tool_call_id=tool_call_outcome.call.id,
            ),
            "interrupted": True,
            "result": tool_call_outcome.result.model_dump(),
            **terminal_controls,
            **copy_tool_round_identity(tool_round_identity).payload(),
        },
    )


def _interrupted_tool_round_results(
    *,
    tool_calls: list[runtime_records.ToolCallRequest],
    completed_outcomes: list[runtime_records.ToolCallOutcome],
    tool_round_identity: ToolRoundIdentity,
    registered_agent: runtime_records.RegisteredAgentState | None = None,
    isolated_dispatched_ids: set[str] | None = None,
    cancellation_artifacts: list[dict[str, Any]] | None = None,
    cancellation_artifacts_by_id: dict[str, list[dict[str, Any]]] | None = None,
) -> list[runtime_records.ToolCallOutcome]:
    completed_ids = {outcome.call.id for outcome in completed_outcomes}
    artifacts_for_interrupted_tool = (
        [] if cancellation_artifacts is None else cancellation_artifacts
    )
    interrupted_outcomes: list[runtime_records.ToolCallOutcome] = []
    for tool_call in tool_calls:
        if tool_call.id in completed_ids:
            continue
        if cancellation_artifacts_by_id is not None:
            result_artifacts = cancellation_artifacts_by_id.get(tool_call.id, [])
        else:
            result_artifacts = artifacts_for_interrupted_tool
            artifacts_for_interrupted_tool = []
        interrupted_outcomes.append(
            _interrupted_tool_call_outcome(
                tool_call=tool_call,
                tool_round_identity=tool_round_identity,
                registered_tool=(
                    None
                    if registered_agent is None
                    else registered_agent.executable_tool(tool_call.name)
                ),
                execution_started=(
                    isolated_dispatched_ids is not None and tool_call.id in isolated_dispatched_ids
                ),
                artifacts=result_artifacts,
            )
        )
    return interrupted_outcomes
