"""Paused-round staging owned by DurableToolRound.

Approval and input policy supply decisions. This phase owns capacity restoration,
secret sealing, restarted-stage fencing and ordered terminal publication.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from contextlib import aclosing
from datetime import datetime

from cayu.events import Event, EventType, event_with_runtime_payload_authority
from cayu.runtime import _invocation_secrets as invocation_secrets
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _tool_execution as tool_execution
from cayu.runtime import _tool_results as tool_results
from cayu.runtime import _tool_round_recovery as tool_round_recovery
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime._invocation_lifecycle import InvocationContext
from cayu.runtime._tool_round_staging import (
    ToolTerminalPublisher,
    _staged_terminal_argument_projections,
    _ToolRoundPublicationCoordinator,
)
from cayu.runtime.execution_profiles import (
    ExecutionProfileIdentity,
    event_with_execution_profile_authority,
)
from cayu.runtime.execution_units import ToolRoundIdentity
from cayu.sessions.base import Session, SessionStore
from cayu.tools._redaction import InvocationRedactorSnapshot
from cayu.tools.base import ToolResult
from cayu.tools.exposure import ResolvedToolExposureAuthority
from cayu.tools.terminal_publication import ToolTerminalPublicationGovernor
from cayu.vaults.redaction import SecretRedactor


class ToolRoundContinuation:
    """Private continuation phase; callers use the durable round owner."""

    def __init__(
        self,
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
    ) -> None:
        self._session = session
        self._identity = tool_round_identity
        self._session_store = session_store
        self._event_writer = event_writer
        self._registered_agent = registered_agent
        self._registered_environment = registered_environment
        self._environment_name = environment_name
        self._tool_calls = tool_calls
        self._task_id = task_id
        self._execution_profile = execution_profile
        self._invocation_context = invocation_context
        self._base_redactor = redactor
        self._publication_governor = publication_governor
        self._clock = clock
        self._emit_result = emit_result
        self._emit_terminal = emit_terminal
        self._pause_authority = pause_authority
        self._idempotency_options = idempotency_options
        self._hook_modes: dict[str, tuple[bool, bool]] = {}
        self._admitted = False
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

    @property
    def redactor(self) -> SecretRedactor | None:
        return None if self._coordinator is None else self._coordinator.redactor

    async def admit(self) -> None:
        if self._admitted:
            raise RuntimeError("Tool round is already admitted.")
        if self._coordinator is not None:
            await self._coordinator.reserve_capacity()
            await self._coordinator.restore_staged_capacity(
                tool_round_recovery.checkpoint_staged_terminals(
                    await self._session_store.load_checkpoint(self._session.id),
                    tool_round_identity=self._identity,
                )
            )
        self._admitted = True

    def finish_dispatch(self) -> None:
        if self._coordinator is not None:
            self._coordinator.seal_capacity()

    def _require_admitted(self) -> _ToolRoundPublicationCoordinator | None:
        if not self._admitted:
            raise RuntimeError("Tool-round publication requires admission.")
        return self._coordinator

    async def record_publication_snapshot(
        self,
        tool_call_id: str,
        snapshot: invocation_secrets.InvocationPublicationSnapshot,
    ) -> None:
        coordinator = self._require_admitted()
        if coordinator is not None:
            await coordinator.seal_call(
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
        coordinator = self._require_admitted()
        if coordinator is None:
            raise AssertionError("Continuation redactor observer has no coordinator.")
        await coordinator.register_redactor(
            tool_call_id=tool_call_id,
            redactor=snapshot.redactor,
        )

    async def stage_terminal(
        self,
        event: Event,
        outcome: runtime_records.ToolCallOutcome,
        allow_modification: bool,
        publish_before_hooks: bool,
        snapshot: invocation_secrets.InvocationPublicationSnapshot,
    ) -> Event:
        coordinator = self._require_admitted()
        if coordinator is None:
            raise AssertionError("Continuation terminal staging has no coordinator.")
        prepared_event = self._event_writer.prepare(event)
        exposure_blocked = (
            prepared_event.type is EventType.TOOL_CALL_BLOCKED
            and prepared_event.payload.get("blocked_by") == "tool_exposure"
        )
        staged = await coordinator.stage_terminal(
            tool_call_id=outcome.call.id,
            event=prepared_event,
            snapshot=snapshot,
            hooks_state=(
                "completed"
                if exposure_blocked
                else (
                    "observational"
                    if publish_before_hooks
                    else ("pending" if allow_modification else "finalized")
                )
            ),
        )
        self._hook_modes[outcome.call.id] = (
            allow_modification,
            publish_before_hooks,
        )
        return staged

    async def record_workspace_capture(self, event: Event) -> Event:
        coordinator = self._require_admitted()
        if coordinator is None:
            raise AssertionError("Workspace capture recording has no coordinator.")
        return await coordinator.record_workspace_capture(event)

    async def record_static_scope(
        self,
        tool_call_id: str,
        *,
        execution_scope_unknown: bool = False,
    ) -> None:
        await self.record_publication_snapshot(
            tool_call_id,
            invocation_secrets.InvocationPublicationSnapshot(
                redactor=self._base_redactor,
                unsafe_output=execution_scope_unknown,
                secret_scope_incomplete=execution_scope_unknown,
            ),
        )

    async def fence_restarted(
        self,
        *,
        recorded_ids: set[str],
        resume_undispatched_siblings: bool = False,
    ) -> set[str]:
        """Close a partially staged continuation without re-executing siblings."""
        coordinator = self._require_admitted()
        if coordinator is None:
            return set()

        checkpoint = await self._session_store.load_checkpoint(self._session.id)
        stages = tool_round_recovery.checkpoint_staged_terminals(
            checkpoint,
            tool_round_identity=coordinator.tool_round_identity,
        )
        if not stages:
            return set()
        staged_ids = {item.tool_call_id for item in stages}
        for staged in stages:
            if staged.tool_call_id in recorded_ids or staged.hooks_state == "completed":
                continue
            if (
                resume_undispatched_siblings
                and staged.staged_at is not None
                and staged.publication_started_at is None
            ):
                # The owned stage was never published or exposed to terminal
                # hooks. With static scope, normal publication can still run
                # those hooks once without replacing the accepted input result.
                continue
            unavailable = tool_round_recovery.hook_scope_unavailable_recovery_event(staged.event)
            await self._session_store.transform_checkpoint(
                self._session.id,
                tool_round_recovery.completed_staged_terminal_transform(
                    tool_round_identity=coordinator.tool_round_identity,
                    event=unavailable,
                ),
            )

        if resume_undispatched_siblings:
            # The exact delegated gate has a positively static secret scope.
            # Existing stages still receive the conservative hook projection
            # above; remaining calls re-enter their normal effect/dispatch owners.
            # Dynamic or unknown scopes never take this continuation entrance.
            return staged_ids

        for tool_call in self._tool_calls:
            if tool_call.id in recorded_ids or tool_call.id in staged_ids:
                continue
            result = ToolResult(
                content=(
                    "Tool call was not executed because recovery could not reconstruct "
                    "the complete sibling invocation-secret scope."
                ),
                structured={
                    "error": "invalid_tool_output",
                    "executed": False,
                    "outcome_unknown": False,
                    "recovered": True,
                    "reason": "continuation_secret_scope_unavailable",
                },
                is_error=True,
            )
            event = event_with_execution_profile_authority(
                Event(
                    type=EventType.TOOL_CALL_BLOCKED,
                    session_id=self._session.id,
                    agent_name=self._registered_agent.spec.name,
                    environment_name=self._environment_name,
                    tool_name=tool_call.name,
                    payload={
                        **coordinator.tool_round_identity.payload(),
                        **self._pause_authority,
                        "tool_call_id": tool_call.id,
                        "idempotency_key": tool_execution.tool_idempotency_key(
                            session_id=self._session.id,
                            tool_round_id=coordinator.tool_round_identity.tool_round_id,
                            tool_call_id=tool_call.id,
                            **self._idempotency_options,
                        ),
                        "recovered": True,
                        "result": result.model_dump(mode="json"),
                    },
                ),
                self._execution_profile,
            )
            staged_event = await coordinator.stage_terminal(
                tool_call_id=tool_call.id,
                event=self._event_writer.prepare(event),
                snapshot=invocation_secrets.InvocationPublicationSnapshot(
                    redactor=coordinator.redactor,
                    unsafe_output=False,
                    secret_scope_incomplete=False,
                ),
                hooks_state="finalized",
            )
            await coordinator.complete_terminal_hooks(staged_event)
            staged_ids.add(tool_call.id)
        return staged_ids

    async def publish(
        self,
        *,
        already_published_ids: set[str],
        restarted_staged_ids: set[str],
    ) -> AsyncGenerator[tuple[Event, runtime_records.ToolCallOutcome | None], None]:
        """Publish continuation results only after the round scope is final."""
        coordinator = self._require_admitted()
        if coordinator is None:
            return
        expected_staged_ids = {
            call.id for call in self._tool_calls if call.id not in already_published_ids
        } | (already_published_ids & restarted_staged_ids)
        current_stages = tool_round_recovery.checkpoint_staged_terminals(
            await self._session_store.load_checkpoint(self._session.id),
            tool_round_identity=self._identity,
        )
        if {item.tool_call_id for item in current_stages} != expected_staged_ids:
            kind = "user-input" if "input_id" in self._pause_authority else "approval"
            raise RuntimeError(
                f"Dynamic {kind} continuation requires one private terminal "
                "stage per unresolved call."
            )
        identity = coordinator.tool_round_identity
        checkpoint = await self._session_store.load_checkpoint(self._session.id)
        staged_records = tool_round_recovery.checkpoint_staged_terminals(
            checkpoint,
            tool_round_identity=identity,
        )
        staged_by_id = {item.tool_call_id: item for item in staged_records}
        calls_by_id = {call.id: call for call in self._tool_calls}
        if not set(staged_by_id).issubset(calls_by_id):
            raise RuntimeError("Continuation stages contain a call outside their tool round.")
        if not already_published_ids.issubset(calls_by_id):
            raise RuntimeError("Published continuation evidence names an unknown tool call.")

        async def complete_hooks(event: Event) -> Event:
            return await coordinator.complete_terminal_hooks(event)

        async def record_projection(event: Event) -> Event:
            return await coordinator.record_projected_terminal(event)

        for tool_call in self._tool_calls:
            staged = staged_by_id.get(tool_call.id)
            if staged is None:
                continue
            if tool_call.id in already_published_ids:
                self._publication_governor.published(
                    session_id=self._session.id,
                    event_id=staged.event.id,
                    published_at=self._clock(),
                )
                coordinator.terminal_published(staged.event.id)
                continue
            staged = await coordinator.start_publication(staged)
            staged_event = await self._publication_governor.run_cpu(
                staged.payload_bytes or 0,
                lambda staged=staged: coordinator.restore_started_publication_authority(staged),
            )
            authority_fields: list[str] = []
            for field_name, expected_value in self._pause_authority.items():
                if staged_event.payload.get(field_name) != expected_value:
                    raise RuntimeError(
                        "Continuation stage conflicts with its pending pause identity."
                    )
                authority_fields.append(field_name)
            if authority_fields:
                staged_event = event_with_runtime_payload_authority(
                    staged_event,
                    *authority_fields,
                )
            result_payload = staged_event.payload.get("result")
            if type(result_payload) is not dict:
                raise RuntimeError("Continuation stage lost its tool result.")
            result = tool_results.tool_result_from_payload(result_payload)
            argument_projection, hook_argument_projection = _staged_terminal_argument_projections(
                staged_event
            )
            hooks_already_completed = staged.hooks_state == "completed"
            allow_modification, publish_before_hooks = (
                (False, False)
                if hooks_already_completed
                else self._hook_modes.get(
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
                    record_projection
                    if publish_before_hooks and not hooks_already_completed
                    else None
                ),
                deferred_terminal_finalizer=(None if hooks_already_completed else complete_hooks),
                terminal_event_emitter=self._emit_terminal,
                hooks_already_completed=hooks_already_completed,
            )
            async with aclosing(terminal_stream) as terminal_events:
                async for event, outcome in terminal_events:
                    if event.type in tool_round_recovery._TOOL_ROUND_TERMINAL_EVENT_TYPES:
                        self._publication_governor.published(
                            session_id=self._session.id,
                            event_id=event.id,
                            published_at=self._clock(),
                        )
                        coordinator.terminal_published(event.id)
                    yield event, outcome
        coordinator.seal_capacity()
