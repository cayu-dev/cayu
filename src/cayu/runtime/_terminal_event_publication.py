"""Publish and replay terminal events with exactly-once runtime hooks."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, cast
from uuid import UUID, uuid4

from cayu._exception_groups import (
    iter_exception_tree,
)
from cayu._validation import (
    canonical_durable_json_bytes,
    copy_durable_json_value,
    require_clean_nonblank,
)
from cayu.events import (
    Event,
    EventType,
    copy_event,
    event_id_is_runtime_generated,
    event_with_runtime_envelope_authority,
    event_with_runtime_generated_id,
    event_with_runtime_payload_authority,
)
from cayu.exceptions import (
    TerminalEventPublicationUncertain,
)
from cayu.execution_profiles import (
    ExecutionProfileIdentity,
    event_with_execution_profile_authority,
)
from cayu.observability.hooks import (
    RuntimeHook,
    RuntimeHookContext,
    RuntimeHookPhase,
    RuntimeHookRuntime,
    _runtime_hook_supports_phase,
)
from cayu.observability.hooks import (
    _runtime_hook_event as _build_runtime_hook_event,
)
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime._diagnostics import (
    exception_diagnostic,
)
from cayu.runtime._durable_tool_round import _environment_name
from cayu.runtime._environment_lifecycle import (
    EnvironmentBindingFinalizeResult,
    EnvironmentLifecycle,
)
from cayu.runtime._event_writer import (
    RuntimeEventWriter,
    _reconcile_exact_persisted_event,
)
from cayu.runtime._invocation_lifecycle import (
    InvocationContext,
)
from cayu.runtime._recovery_requests import (
    RecoveryTerminalEventRequest,
)
from cayu.sessions._terminal_evidence import (
    _session_run_operation_from_checkpoint,
)
from cayu.sessions.base import (
    _SESSION_RUN_OPERATION_ID_PAYLOAD_KEY,
    SessionRuntimePublicationConflict,
    SessionStore,
    _checkpoint_after_session_run_operation_cleanup,
    _event_with_session_run_operation,
    _mark_session_invocation_terminal_event,
)
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.records import (
    Session,
    SessionStatus,
)
from cayu.vaults.redaction import SecretRedactor
from cayu.workspaces.observation_recovery import (
    retain_workspace_observation_pending_cancellation_requests,
)

logger = logging.getLogger(__name__)


class _TerminalRuntimeHookClaimState(StrEnum):
    CLAIMED = "claimed"
    SETTLED = "settled"
    IN_PROGRESS = "in_progress"


@dataclass(frozen=True)
class _TerminalRuntimeHookClaim:
    state: _TerminalRuntimeHookClaimState
    started_event: Event | None = None

    def __post_init__(self) -> None:
        if self.state is _TerminalRuntimeHookClaimState.CLAIMED:
            if type(self.started_event) is not Event:
                raise TypeError("A claimed terminal runtime hook requires its started event.")
        elif self.started_event is not None:
            raise ValueError("An unowned terminal runtime hook cannot expose a started event.")


async def _call_runtime_hook(
    *,
    hook: RuntimeHook,
    phase: RuntimeHookPhase,
    context: RuntimeHookContext,
) -> None:
    if phase == RuntimeHookPhase.AFTER_SESSION_COMPLETED:
        await hook.after_session_completed(context)
        return
    if phase == RuntimeHookPhase.AFTER_SESSION_FAILED:
        await hook.after_session_failed(context)
        return
    if phase == RuntimeHookPhase.AFTER_SESSION_INTERRUPTED:
        await hook.after_session_interrupted(context)
        return
    raise ValueError(f"Unsupported runtime hook phase: {phase}")


def _runtime_hook_actions_payload(context: RuntimeHookContext) -> dict[str, Any]:
    """Return portable action evidence without making hooks unterminalizable."""

    try:
        actions = copy_durable_json_value(context.actions, "hook_actions")
    except BaseException:
        return {"actions": [], "actions_omitted": True}
    if type(actions) is not list:
        return {"actions": [], "actions_omitted": True}
    return {"actions": actions}


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
    return event_with_execution_profile_authority(
        _build_runtime_hook_event(
            event_type=event_type,
            hook_name=hook_name,
            scope=scope,
            phase=phase,
            session=session,
            terminal_event=terminal_event,
            agent_name=registered_agent.spec.name,
            environment_name=_environment_name(registered_environment),
            payload=payload,
        ),
        execution_profile,
    )


def _terminal_runtime_hook_event_id(
    *,
    session: Session,
    terminal_event: Event,
    phase: RuntimeHookPhase,
    scope: str,
    hook_index: int,
    outcome: Literal["started", "completed", "failed"],
) -> str:
    """Return one content-addressed lifecycle identity for a terminal hook slot."""

    if not isinstance(phase, RuntimeHookPhase):
        raise TypeError("phase must be a RuntimeHookPhase.")
    if type(hook_index) is not int or hook_index < 0:
        raise ValueError("hook_index must be a non-negative integer.")
    scope = require_clean_nonblank(scope, "runtime_hook.scope")
    if scope not in {"app", "agent"}:
        raise ValueError("Terminal runtime hook scope must be 'app' or 'agent'.")
    material = canonical_durable_json_bytes(
        {
            "schema": "cayu.terminal-runtime-hook.v1",
            "session_id": require_clean_nonblank(session.id, "session.id"),
            "terminal_event_id": require_clean_nonblank(
                terminal_event.id,
                "terminal_event.id",
            ),
            "terminal_event_type": str(terminal_event.type),
            "phase": phase.value,
            "scope": scope,
            "hook_index": hook_index,
        },
        "terminal_runtime_hook_event_id",
    )
    digest = hashlib.sha256(material).hexdigest()
    return f"terminal-runtime-hook:v1:{digest}:{outcome}"


def _terminal_runtime_hook_event(
    *,
    event_type: EventType,
    outcome: Literal["started", "completed", "failed"],
    hook_invocation_id: str,
    hook_index: int,
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
    """Build one claim-bound terminal-hook lifecycle event."""

    expected_event_type = {
        "started": EventType.HOOK_STARTED,
        "completed": EventType.HOOK_COMPLETED,
        "failed": EventType.HOOK_FAILED,
    }[outcome]
    if event_type != expected_event_type:
        raise ValueError("Terminal runtime hook outcome does not match its event type.")
    raw_hook_invocation_id = require_clean_nonblank(
        hook_invocation_id,
        "hook_invocation_id",
    )
    parsed_hook_invocation_id = UUID(raw_hook_invocation_id)
    if (
        parsed_hook_invocation_id.version != 4
        or str(parsed_hook_invocation_id) != raw_hook_invocation_id
    ):
        raise ValueError("hook_invocation_id must be a canonical UUID4.")
    if type(hook_index) is not int or hook_index < 0:
        raise ValueError("hook_index must be a non-negative integer.")
    event = _runtime_hook_event(
        event_type=event_type,
        hook_name=hook_name,
        scope=scope,
        phase=phase,
        session=session,
        registered_agent=registered_agent,
        registered_environment=registered_environment,
        terminal_event=terminal_event,
        payload={
            **payload,
            "hook_index": hook_index,
            "hook_invocation_id": raw_hook_invocation_id,
        },
        execution_profile=execution_profile,
    ).model_copy(
        update={
            "id": _terminal_runtime_hook_event_id(
                session=session,
                terminal_event=terminal_event,
                phase=phase,
                scope=scope,
                hook_index=hook_index,
                outcome=outcome,
            ),
            # A terminal hook belongs to the terminal event's logical
            # interaction, not to whichever concurrent continuation context won
            # the durable reservation.
            "interaction_id": terminal_event.interaction_id,
        }
    )
    if event.interaction_id is not None:
        event = event_with_runtime_envelope_authority(event, "interaction_id")
    return event_with_runtime_payload_authority(
        event_with_runtime_generated_id(event),
        "hook_invocation_id",
    )


def _terminal_runtime_hook_invocation_id(event: Event) -> str:
    raw = event.payload.get("hook_invocation_id")
    if type(raw) is not str:
        raise ValueError("hook_invocation_id must be a string.")
    raw = require_clean_nonblank(raw, "hook_invocation_id")
    parsed = UUID(raw)
    if parsed.version != 4 or str(parsed) != raw:
        raise ValueError("hook_invocation_id must be a canonical UUID4.")
    return raw


def _terminal_runtime_hook_started_matches(
    existing: Event,
    expected: Event,
) -> bool:
    """Accept the same hook slot while allowing only its claimant and timestamp to differ."""

    if existing.type != EventType.HOOK_STARTED or expected.type != EventType.HOOK_STARTED:
        return False
    existing_envelope = existing.model_dump(mode="json", exclude={"payload", "timestamp"})
    expected_envelope = expected.model_dump(mode="json", exclude={"payload", "timestamp"})
    if existing_envelope != expected_envelope:
        return False
    existing_payload = dict(existing.payload)
    expected_payload = dict(expected.payload)
    try:
        _terminal_runtime_hook_invocation_id(existing)
        _terminal_runtime_hook_invocation_id(expected)
    except (TypeError, ValueError):
        return False
    existing_payload.pop("hook_invocation_id", None)
    expected_payload.pop("hook_invocation_id", None)
    return existing_payload == expected_payload


def _terminal_runtime_hook_outcome_matches(
    outcome: Event,
    *,
    started: Event,
    expected_event_id: str,
    expected_type: EventType,
) -> bool:
    """Authenticate a completed/failed marker against its durable reservation."""

    if outcome.id != expected_event_id or outcome.type != expected_type:
        return False
    if any(
        getattr(outcome, field_name) != getattr(started, field_name)
        for field_name in (
            "session_id",
            "interaction_id",
            "agent_name",
            "environment_name",
            "workflow_name",
            "tool_name",
        )
    ):
        return False
    try:
        invocation_id = _terminal_runtime_hook_invocation_id(started)
        if _terminal_runtime_hook_invocation_id(outcome) != invocation_id:
            return False
    except (TypeError, ValueError):
        return False
    return all(outcome.payload.get(key) == value for key, value in started.payload.items())


class TerminalEventPublication:
    """Publish and replay terminal events with exactly-once runtime hooks."""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        environment_lifecycle: EnvironmentLifecycle,
        event_writer: RuntimeEventWriter,
        secret_redactor: SecretRedactor,
        runtime_hooks: tuple[runtime_records.RegisteredRuntimeHook, ...],
        hook_runtime: RuntimeHookRuntime,
    ) -> None:
        self.session_store = session_store
        self._environment_lifecycle = environment_lifecycle
        self._event_writer = event_writer
        self._secret_redactor = secret_redactor
        self._runtime_hooks = runtime_hooks
        self._hook_runtime = hook_runtime

    async def bind_to_run_operation(
        self,
        event: Event,
        *,
        session: Session,
        expected_run_operation_epoch: int | None = None,
    ) -> Event:
        checkpoint = await self.session_store.load_checkpoint(session.id)
        run_operation = _session_run_operation_from_checkpoint(checkpoint)
        if run_operation is None:
            return event
        expected_run_epoch = session.run_epoch
        if expected_run_operation_epoch is not None:
            if session.status is not SessionStatus.FAILED or expected_run_operation_epoch not in {
                session.run_epoch,
                session.run_epoch - 1,
            }:
                raise RuntimeError("Terminal event cannot adopt an unrelated released run epoch.")
            expected_run_epoch = expected_run_operation_epoch
        if run_operation.run_epoch != expected_run_epoch:
            raise RuntimeError(
                "Terminal event session run operation does not match the active run epoch."
            )
        return _event_with_session_run_operation(event, run_operation)

    async def clear_run_operation(
        self,
        *,
        session_id: str,
        operation_id: str,
        terminal_evidence_durable: bool = False,
    ) -> None:
        def clear_operation(
            _session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any] | None:
            run_operation = _session_run_operation_from_checkpoint(checkpoint)
            if run_operation is None:
                return checkpoint
            if run_operation.operation_id != operation_id:
                raise RuntimeError(
                    "Session run operation changed before terminal evidence cleanup."
                )
            return _checkpoint_after_session_run_operation_cleanup(
                checkpoint,
                operation=run_operation,
                retain_terminal_receipt=terminal_evidence_durable,
            )

        await self.session_store.transform_checkpoint(
            session_id,
            clear_operation,
        )

    async def clear_run_operation_after_terminal_event(
        self,
        *,
        session: Session,
        terminal_event: Event,
    ) -> None:
        """Best-effort trailing cleanup after authoritative terminal evidence."""

        run_operation_id = terminal_event.payload.get(_SESSION_RUN_OPERATION_ID_PAYLOAD_KEY)
        if run_operation_id is None:
            return
        try:
            await self.clear_run_operation(
                session_id=session.id,
                operation_id=require_clean_nonblank(
                    run_operation_id,
                    "terminal event session_run_operation_id",
                ),
                terminal_evidence_durable=True,
            )
        except Exception as cleanup_failure:
            logger.warning(
                "Terminal evidence is durable but session run operation cleanup "
                "remains pending: session_id=%s event_id=%s error_type=%s",
                session.id,
                terminal_event.id,
                type(cleanup_failure).__name__,
            )

    async def reconcile_persisted_terminal_event(
        self,
        event: Event,
    ) -> Event | None:
        """Return the exact durable event after an ambiguous publication failure."""
        records = await self.session_store.query_events(
            EventQuery(
                session_id=event.session_id,
                event_id=event.id,
                limit=1,
            )
        )
        return _reconcile_exact_persisted_event(
            event,
            records,
            conflict_message=(
                "Terminal event identity is already used by different durable evidence."
            ),
        )

    async def emit(
        self,
        *,
        event: Event,
        phase: RuntimeHookPhase,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        execution_profile: ExecutionProfileIdentity | None = None,
        invocation_context: InvocationContext | None = None,
        terminal_event_publisher: Callable[[Event], Awaitable[Event]] | None = None,
        expected_run_operation_epoch: int | None = None,
        run_runtime_hooks: bool = True,
        environment_finalize_result: EnvironmentBindingFinalizeResult | None = None,
    ) -> AsyncGenerator[Event, None]:
        if invocation_context is not None:
            if (
                invocation_context.binding.session_id != session.id
                or registered_agent is not invocation_context.registered_agent
                or registered_environment is not invocation_context.registered_environment
            ):
                raise RuntimeError("Terminal publication substituted frozen invocation authority.")
            if (
                execution_profile is not None
                and execution_profile is not invocation_context.profile
            ):
                raise RuntimeError("Terminal publication substituted its execution profile.")
            execution_profile = invocation_context.profile
        if environment_finalize_result is None:
            event = await self.bind_to_run_operation(
                event,
                session=session,
                expected_run_operation_epoch=expected_run_operation_epoch,
            )
            finalize_result = await self._environment_lifecycle.finalize_terminal_event(
                event=event,
                session=session,
                registered_environment=registered_environment,
                execution_profile=execution_profile,
                invocation_context=invocation_context,
            )
        else:
            finalize_result = environment_finalize_result
            if finalize_result.event.model_dump(mode="json") != event.model_dump(mode="json"):
                raise RuntimeError(
                    "Pre-finalized environment result belongs to a different terminal event."
                )

        async def emit_finalized_terminal_boundary() -> AsyncGenerator[Event, None]:
            for binding_event in finalize_result.events:
                yield binding_event
            prepared_terminal_event = self._event_writer.prepare(
                event_with_runtime_envelope_authority(
                    finalize_result.event,
                    "session_id",
                )
            )
            try:
                terminal_event = await (
                    self._event_writer.emit(prepared_terminal_event)
                    if terminal_event_publisher is None
                    else terminal_event_publisher(prepared_terminal_event)
                )
            except Exception as publication_failure:
                try:
                    reconciled_terminal_event = await self.reconcile_persisted_terminal_event(
                        prepared_terminal_event
                    )
                except Exception as reconciliation_failure:
                    uncertainty = TerminalEventPublicationUncertain(
                        event=prepared_terminal_event,
                        publication_failure=publication_failure,
                        reconciliation_failure=reconciliation_failure,
                    )
                    raise uncertainty from uncertainty.failures
                if reconciled_terminal_event is None:
                    raise
                terminal_event = reconciled_terminal_event
                logger.warning(
                    "Terminal event is durable but its publication acknowledgement or "
                    "side-effect delivery failed: session_id=%s event_id=%s error_type=%s",
                    session.id,
                    terminal_event.id,
                    type(publication_failure).__name__,
                )
            if terminal_event.model_dump(mode="json") != prepared_terminal_event.model_dump(
                mode="json"
            ):
                raise RuntimeError(
                    "Terminal event publication returned different durable evidence."
                )
            if event_id_is_runtime_generated(prepared_terminal_event):
                terminal_event = event_with_runtime_generated_id(terminal_event)
            _mark_session_invocation_terminal_event(terminal_event)
            await self.clear_run_operation_after_terminal_event(
                session=session,
                terminal_event=terminal_event,
            )
            yield terminal_event
            if not run_runtime_hooks:
                return
            hook_stream = self._run_runtime_hooks(
                phase=phase,
                session=session,
                terminal_event=terminal_event,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                hooks=self._ordered_terminal_runtime_hooks(
                    registered_agent=registered_agent,
                    invocation_context=invocation_context,
                ),
                execution_profile=execution_profile,
                invocation_context=invocation_context,
            )
            async with contextlib.aclosing(hook_stream) as owned_hook_stream:
                async for hook_event in owned_hook_stream:
                    yield hook_event

        try:
            async with contextlib.aclosing(emit_finalized_terminal_boundary()) as terminal_stream:
                async for emitted_event in terminal_stream:
                    yield emitted_event
        except BaseException as post_finalize_failure:
            cancellation = finalize_result.cancellation
            if cancellation is None or any(
                candidate is cancellation
                for candidate in iter_exception_tree(post_finalize_failure)
            ):
                raise
            if isinstance(post_finalize_failure, Exception):
                if finalize_result.cancellation_requests_consumed:
                    retain_workspace_observation_pending_cancellation_requests(
                        cancellation,
                        finalize_result.cancellation_requests_consumed,
                    )
                raise cancellation from post_finalize_failure
            concurrent_control = BaseExceptionGroup(
                "Terminal finalization received concurrent control after egress parking.",
                [post_finalize_failure, cancellation],
            )
            if finalize_result.cancellation_requests_consumed:
                retain_workspace_observation_pending_cancellation_requests(
                    concurrent_control,
                    finalize_result.cancellation_requests_consumed,
                )
            raise concurrent_control from None

        if finalize_result.cancellation is not None:
            if finalize_result.cancellation_requests_consumed:
                retain_workspace_observation_pending_cancellation_requests(
                    finalize_result.cancellation,
                    finalize_result.cancellation_requests_consumed,
                )
            raise finalize_result.cancellation

    def _ordered_terminal_runtime_hooks(
        self,
        *,
        registered_agent: runtime_records.RegisteredAgentState,
        invocation_context: InvocationContext | None,
    ) -> tuple[tuple[runtime_records.RegisteredRuntimeHook, str, int], ...]:
        app_hooks = (
            self._runtime_hooks if invocation_context is None else invocation_context.runtime_hooks
        )
        return (
            *((registered_hook, "app", index) for index, registered_hook in enumerate(app_hooks)),
            *(
                (registered_hook, "agent", index)
                for index, registered_hook in enumerate(registered_agent.runtime_hooks)
            ),
        )

    async def replay(
        self,
        *,
        event: Event,
        phase: RuntimeHookPhase,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        execution_profile: ExecutionProfileIdentity | None = None,
        invocation_context: InvocationContext | None = None,
        run_runtime_hooks: bool = True,
        yield_terminal_event: bool = True,
    ) -> AsyncGenerator[Event, None]:
        """Replay exact terminal evidence and converge any unclaimed hook slots."""

        if invocation_context is not None:
            if (
                invocation_context.binding.session_id != session.id
                or registered_agent is not invocation_context.registered_agent
                or registered_environment is not invocation_context.registered_environment
            ):
                raise RuntimeError("Terminal replay substituted frozen invocation authority.")
            if (
                execution_profile is not None
                and execution_profile is not invocation_context.profile
            ):
                raise RuntimeError("Terminal replay substituted its execution profile.")
            execution_profile = invocation_context.profile
        records = await self.session_store.query_events(
            EventQuery(session_id=event.session_id, event_id=event.id, limit=2)
        )
        if len(records) != 1:
            raise SessionRuntimePublicationConflict(
                "Terminal replay requires exactly one durable terminal event."
            )
        durable_event = records[0].event
        if durable_event.model_dump(mode="json") != event.model_dump(mode="json"):
            raise SessionRuntimePublicationConflict(
                "Terminal replay evidence conflicts with its durable event."
            )
        _mark_session_invocation_terminal_event(durable_event)
        await self.clear_run_operation_after_terminal_event(
            session=session,
            terminal_event=durable_event,
        )
        if yield_terminal_event:
            yield copy_event(durable_event)
        if not run_runtime_hooks:
            return
        async for hook_event in self._run_runtime_hooks(
            phase=phase,
            session=session,
            terminal_event=durable_event,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            hooks=self._ordered_terminal_runtime_hooks(
                registered_agent=registered_agent,
                invocation_context=invocation_context,
            ),
            execution_profile=execution_profile,
            invocation_context=invocation_context,
        ):
            yield hook_event

    async def _load_terminal_runtime_hook_settlement(
        self,
        *,
        started_event: Event,
        completed_event_id: str,
        failed_event_id: str,
    ) -> bool:
        """Return whether one exact reservation has one authenticated terminal marker."""

        outcomes: list[tuple[Event, EventType, str]] = []
        for event_id, event_type in (
            (completed_event_id, EventType.HOOK_COMPLETED),
            (failed_event_id, EventType.HOOK_FAILED),
        ):
            records = await self.session_store.query_events(
                EventQuery(
                    session_id=started_event.session_id,
                    event_id=event_id,
                    limit=2,
                )
            )
            if len(records) > 1:
                raise SessionRuntimePublicationConflict(
                    "Terminal runtime hook has duplicate outcome evidence."
                )
            if records:
                outcomes.append((records[0].event, event_type, event_id))
        if not outcomes:
            return False
        if len(outcomes) != 1:
            raise SessionRuntimePublicationConflict(
                "Terminal runtime hook has conflicting completed and failed outcomes."
            )
        outcome, expected_type, expected_event_id = outcomes[0]
        if not _terminal_runtime_hook_outcome_matches(
            outcome,
            started=started_event,
            expected_event_id=expected_event_id,
            expected_type=expected_type,
        ):
            raise SessionRuntimePublicationConflict(
                "Terminal runtime hook outcome conflicts with its durable reservation."
            )
        return True

    async def _claim_terminal_runtime_hook(
        self,
        *,
        started_event: Event,
        completed_event_id: str,
        failed_event_id: str,
    ) -> _TerminalRuntimeHookClaim:
        """Atomically reserve one terminal hook or classify its durable peer owner."""

        prepared = self._event_writer.prepare(started_event)
        requested_invocation_id = _terminal_runtime_hook_invocation_id(prepared)
        try:
            await self.session_store.append_event(prepared.session_id, prepared)
        except Exception as append_failure:
            try:
                records = await self.session_store.query_events(
                    EventQuery(
                        session_id=prepared.session_id,
                        event_id=prepared.id,
                        limit=2,
                    )
                )
            except Exception as verification_failure:
                append_failure.add_note(
                    "Terminal runtime hook reservation verification also failed: "
                    f"{type(verification_failure).__name__}"
                )
                raise append_failure from verification_failure
            if not records:
                raise
            if len(records) != 1:
                raise SessionRuntimePublicationConflict(
                    "Terminal runtime hook has duplicate reservation evidence."
                ) from append_failure
            persisted = records[0].event
            if not _terminal_runtime_hook_started_matches(persisted, prepared):
                raise SessionRuntimePublicationConflict(
                    "Terminal runtime hook reservation identity is already used by "
                    "different durable evidence."
                ) from append_failure
            persisted_invocation_id = _terminal_runtime_hook_invocation_id(persisted)
            if persisted_invocation_id != requested_invocation_id:
                settled = await self._load_terminal_runtime_hook_settlement(
                    started_event=persisted,
                    completed_event_id=completed_event_id,
                    failed_event_id=failed_event_id,
                )
                return _TerminalRuntimeHookClaim(
                    state=(
                        _TerminalRuntimeHookClaimState.SETTLED
                        if settled
                        else _TerminalRuntimeHookClaimState.IN_PROGRESS
                    )
                )
            if persisted.model_dump(mode="json") != prepared.model_dump(mode="json"):
                raise SessionRuntimePublicationConflict(
                    "Terminal runtime hook claimant has conflicting durable evidence."
                ) from append_failure

        return _TerminalRuntimeHookClaim(
            state=_TerminalRuntimeHookClaimState.CLAIMED,
            started_event=prepared,
        )

    async def _execute_terminal_runtime_hook_slot(
        self,
        *,
        started_event: Event,
        completed_event_id: str,
        failed_event_id: str,
        hook: RuntimeHook,
        hook_name: str,
        hook_index: int,
        scope: str,
        phase: RuntimeHookPhase,
        session: Session,
        terminal_event: Event,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        execution_profile: ExecutionProfileIdentity | None,
        hook_entered: asyncio.Event,
        hook_settled: asyncio.Event,
        outcome_persisted: asyncio.Event,
    ) -> tuple[Event, ...] | None:
        """Claim, invoke, and durably settle one terminal-hook slot.

        The caller shields this operation until the hook has entered. Once the
        hook returns (or raises an ordinary exception), its outcome publication
        is likewise protected from caller cancellation. Persisting both lifecycle
        markers before fan-out leaves sink delivery independently recoverable.
        """

        claim = await self._claim_terminal_runtime_hook(
            started_event=started_event,
            completed_event_id=completed_event_id,
            failed_event_id=failed_event_id,
        )
        if claim.state is _TerminalRuntimeHookClaimState.IN_PROGRESS:
            return None
        if claim.state is _TerminalRuntimeHookClaimState.SETTLED:
            return ()
        if claim.started_event is None:
            raise AssertionError("Claimed terminal runtime hook lost its started event.")

        context = RuntimeHookContext(
            runtime=self._hook_runtime,
            hook_name=hook_name,
            phase=phase,
            session=session,
            terminal_event=terminal_event,
            execution_profile=execution_profile,
        )
        hook_invocation_id = _terminal_runtime_hook_invocation_id(claim.started_event)
        hook_entered.set()
        try:
            await _call_runtime_hook(hook=hook, phase=phase, context=context)
        except Exception as exc:
            diagnostic = exception_diagnostic(
                exc,
                empty_message="runtime hook failed",
                nonportable_message="Runtime hook failed with a non-portable diagnostic.",
                redactor=self._secret_redactor,
            )
            outcome_template = _terminal_runtime_hook_event(
                event_type=EventType.HOOK_FAILED,
                outcome="failed",
                hook_invocation_id=hook_invocation_id,
                hook_index=hook_index,
                hook_name=hook_name,
                scope=scope,
                phase=phase,
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                terminal_event=terminal_event,
                execution_profile=execution_profile,
                payload={
                    **diagnostic.payload_fields(),
                    **_runtime_hook_actions_payload(context),
                },
            )
        else:
            outcome_template = _terminal_runtime_hook_event(
                event_type=EventType.HOOK_COMPLETED,
                outcome="completed",
                hook_invocation_id=hook_invocation_id,
                hook_index=hook_index,
                hook_name=hook_name,
                scope=scope,
                phase=phase,
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                terminal_event=terminal_event,
                execution_profile=execution_profile,
                payload=_runtime_hook_actions_payload(context),
            )
        hook_settled.set()
        persisted_outcome = await self._event_writer.persist_exact_replay(outcome_template)
        outcome_persisted.set()
        delivered = await self._event_writer.fan_out_persisted(
            [claim.started_event, persisted_outcome]
        )
        return tuple(delivered)

    async def _await_terminal_runtime_hook_slot(
        self,
        **kwargs: Any,
    ) -> tuple[Event, ...] | None:
        """Defer cancellation across the two unsafe hook protocol windows."""

        hook_entered = asyncio.Event()
        hook_settled = asyncio.Event()
        outcome_persisted = asyncio.Event()
        operation = asyncio.create_task(
            self._execute_terminal_runtime_hook_slot(
                **kwargs,
                hook_entered=hook_entered,
                hook_settled=hook_settled,
                outcome_persisted=outcome_persisted,
            )
        )
        cancellation: asyncio.CancelledError | None = None
        cancellation_relay: asyncio.Task[None] | None = None

        async def relay_cancellation_when_safe(exc: asyncio.CancelledError) -> None:
            entered_wait = asyncio.create_task(hook_entered.wait())
            persisted_wait = asyncio.create_task(outcome_persisted.wait())
            try:
                while not operation.done():
                    if hook_entered.is_set() and not hook_settled.is_set():
                        operation.cancel(*exc.args)
                        return
                    if outcome_persisted.is_set():
                        operation.cancel(*exc.args)
                        return
                    waiters: set[asyncio.Task[Any]] = {operation}
                    if not hook_entered.is_set():
                        waiters.add(entered_wait)
                    if not outcome_persisted.is_set():
                        waiters.add(persisted_wait)
                    await asyncio.wait(
                        waiters,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
            finally:
                for waiter in (entered_wait, persisted_wait):
                    if not waiter.done():
                        waiter.cancel()
                await asyncio.gather(entered_wait, persisted_wait, return_exceptions=True)

        while not operation.done():
            try:
                await asyncio.shield(operation)
            except asyncio.CancelledError as exc:
                if operation.cancelled():
                    break
                if cancellation is None:
                    cancellation = exc
                    cancellation_relay = asyncio.create_task(relay_cancellation_when_safe(exc))

        missing = object()
        result: tuple[Event, ...] | None | object = missing
        operation_failure: BaseException | None = None
        try:
            result = operation.result()
        except BaseException as exc:
            operation_failure = exc
        if cancellation_relay is not None:
            if not cancellation_relay.done():
                cancellation_relay.cancel()
            await asyncio.gather(cancellation_relay, return_exceptions=True)

        if cancellation is not None:
            if operation_failure is not None and not isinstance(
                operation_failure,
                asyncio.CancelledError,
            ):
                cancellation.add_note(
                    "Terminal runtime hook settlement also failed after cancellation: "
                    f"{type(operation_failure).__name__}."
                )
                raise cancellation from operation_failure
            raise cancellation
        if operation_failure is not None:
            raise operation_failure
        if result is missing:
            raise RuntimeError("Terminal runtime hook operation returned no result.")
        return cast("tuple[Event, ...] | None", result)

    async def _run_runtime_hooks(
        self,
        *,
        phase: RuntimeHookPhase,
        session: Session,
        terminal_event: Event,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        hooks: tuple[tuple[runtime_records.RegisteredRuntimeHook, str, int], ...],
        execution_profile: ExecutionProfileIdentity | None = None,
        invocation_context: InvocationContext | None = None,
    ) -> AsyncGenerator[Event, None]:
        if invocation_context is not None and (
            invocation_context.binding.session_id != session.id
            or invocation_context.registered_agent is not registered_agent
            or invocation_context.registered_environment is not registered_environment
            or invocation_context.profile is not execution_profile
        ):
            raise RuntimeError("Runtime hook execution substituted frozen invocation authority.")
        for registered_hook, scope, hook_index in hooks:
            hook = registered_hook.hook
            if not _runtime_hook_supports_phase(
                hook=hook,
                phase=phase,
            ):
                continue
            hook_name = registered_hook.name
            hook_invocation_id = str(uuid4())
            started_event = _terminal_runtime_hook_event(
                event_type=EventType.HOOK_STARTED,
                outcome="started",
                hook_invocation_id=hook_invocation_id,
                hook_index=hook_index,
                hook_name=hook_name,
                scope=scope,
                phase=phase,
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                terminal_event=terminal_event,
                execution_profile=execution_profile,
                payload={},
            )
            completed_event_id = _terminal_runtime_hook_event_id(
                session=session,
                terminal_event=terminal_event,
                phase=phase,
                scope=scope,
                hook_index=hook_index,
                outcome="completed",
            )
            failed_event_id = _terminal_runtime_hook_event_id(
                session=session,
                terminal_event=terminal_event,
                phase=phase,
                scope=scope,
                hook_index=hook_index,
                outcome="failed",
            )
            hook_events = await self._await_terminal_runtime_hook_slot(
                started_event=started_event,
                completed_event_id=completed_event_id,
                failed_event_id=failed_event_id,
                hook=hook,
                hook_name=hook_name,
                hook_index=hook_index,
                scope=scope,
                phase=phase,
                session=session,
                terminal_event=terminal_event,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                execution_profile=execution_profile,
            )
            if hook_events is None:
                # Preserve app-before-agent and registration ordering. A peer may
                # still own this slot, so a contender cannot skip to later hooks.
                return
            if not hook_events:
                continue
            for hook_event in hook_events:
                yield hook_event

    async def hooks_are_settled(
        self,
        *,
        phase: RuntimeHookPhase,
        session: Session,
        terminal_event: Event,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        execution_profile: ExecutionProfileIdentity | None = None,
        invocation_context: InvocationContext | None = None,
    ) -> bool:
        """Return whether every authenticated terminal-hook slot has settled."""

        if invocation_context is not None:
            if (
                invocation_context.binding.session_id != session.id
                or invocation_context.registered_agent is not registered_agent
                or invocation_context.registered_environment is not registered_environment
            ):
                raise RuntimeError(
                    "Terminal hook settlement inspection substituted frozen authority."
                )
            if (
                execution_profile is not None
                and execution_profile is not invocation_context.profile
            ):
                raise RuntimeError(
                    "Terminal hook settlement inspection substituted its execution profile."
                )
            execution_profile = invocation_context.profile

        for registered_hook, scope, hook_index in self._ordered_terminal_runtime_hooks(
            registered_agent=registered_agent,
            invocation_context=invocation_context,
        ):
            hook = registered_hook.hook
            if not _runtime_hook_supports_phase(hook=hook, phase=phase):
                continue
            started_event_id = _terminal_runtime_hook_event_id(
                session=session,
                terminal_event=terminal_event,
                phase=phase,
                scope=scope,
                hook_index=hook_index,
                outcome="started",
            )
            records = await self.session_store.query_events(
                EventQuery(
                    session_id=session.id,
                    event_id=started_event_id,
                    limit=2,
                )
            )
            if not records:
                return False
            if len(records) != 1:
                raise SessionRuntimePublicationConflict(
                    "Terminal runtime hook has duplicate reservation evidence."
                )
            started = records[0].event
            expected_started = _terminal_runtime_hook_event(
                event_type=EventType.HOOK_STARTED,
                outcome="started",
                hook_invocation_id=_terminal_runtime_hook_invocation_id(started),
                hook_index=hook_index,
                hook_name=registered_hook.name,
                scope=scope,
                phase=phase,
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                terminal_event=terminal_event,
                execution_profile=execution_profile,
                payload={},
            )
            if not _terminal_runtime_hook_started_matches(started, expected_started):
                raise SessionRuntimePublicationConflict(
                    "Terminal runtime hook reservation conflicts with frozen authority."
                )
            completed_event_id = _terminal_runtime_hook_event_id(
                session=session,
                terminal_event=terminal_event,
                phase=phase,
                scope=scope,
                hook_index=hook_index,
                outcome="completed",
            )
            failed_event_id = _terminal_runtime_hook_event_id(
                session=session,
                terminal_event=terminal_event,
                phase=phase,
                scope=scope,
                hook_index=hook_index,
                outcome="failed",
            )
            if not await self._load_terminal_runtime_hook_settlement(
                started_event=started,
                completed_event_id=completed_event_id,
                failed_event_id=failed_event_id,
            ):
                return False
        return True

    def publish_recovered(
        self,
        request: RecoveryTerminalEventRequest,
    ) -> AsyncIterator[Event]:
        if request.terminal_event_already_durable:
            return self.replay(
                event=request.event,
                phase=request.phase,
                session=request.session,
                registered_agent=request.registered_agent,
                registered_environment=request.registered_environment,
                execution_profile=request.execution_profile,
                invocation_context=request.invocation_context,
                run_runtime_hooks=request.run_runtime_hooks,
                yield_terminal_event=request.yield_durable_terminal_event,
            )
        return self.emit(
            event=request.event,
            phase=request.phase,
            session=request.session,
            registered_agent=request.registered_agent,
            registered_environment=request.registered_environment,
            execution_profile=request.execution_profile,
            invocation_context=request.invocation_context,
            run_runtime_hooks=request.run_runtime_hooks,
        )

    async def recovered_hooks_are_settled(
        self,
        request: RecoveryTerminalEventRequest,
    ) -> bool:
        if not request.terminal_event_already_durable:
            raise ValueError("Terminal hook settlement requires a durable terminal event.")
        return await self.hooks_are_settled(
            phase=request.phase,
            session=request.session,
            terminal_event=request.event,
            registered_agent=request.registered_agent,
            registered_environment=request.registered_environment,
            execution_profile=request.execution_profile,
            invocation_context=request.invocation_context,
        )
