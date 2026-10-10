"""Admit a recovered invocation and retain cleanup until handoff settles."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import uuid4

from cayu._validation import (
    copy_durable_metadata,
)
from cayu.budgets._run_limit_accounting import (
    rebase_run_limit_accounting_context,
)
from cayu.budgets.base import (
    BudgetLimit,
    request_budget_limits_for_session,
)
from cayu.budgets.run_limits import RunLimits
from cayu.collaboration.access import CollaborationAccessContext
from cayu.context.structured_output import (
    StructuredOutputSpec,
)
from cayu.context.thinking import ThinkingConfig
from cayu.events import (
    Event,
)
from cayu.execution_profiles import (
    ExecutionProfileIdentity,
)
from cayu.providers.retry_policy import RetryPolicy
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime._environment_lifecycle import (
    EnvironmentLifecycle,
)
from cayu.runtime._invocation_lifecycle import (
    InvocationContext,
)
from cayu.runtime._recovery_ownership import (
    RecoveryOwnership,
)
from cayu.runtime._recovery_requests import (
    RecoverySessionRunRequest,
)
from cayu.runtime._run_limits import (
    SessionUsageTracker,
)
from cayu.runtime._session_control import (
    SessionControl,
)
from cayu.runtime._session_finalization import (
    SessionFinalization,
)
from cayu.runtime._terminal_evidence_finalization import (
    TerminalEvidenceFinalization,
)
from cayu.sessions import _pending_tool_round as pending_rounds
from cayu.sessions._execution_profile_checkpoint import (
    ActiveInvocationExecutionProfile,
    active_invocation_execution_profile_from_checkpoint,
    checkpoint_with_active_invocation_execution_profile,
)
from cayu.sessions.base import (
    CheckpointTransform,
    SessionRunFenced,
    SessionStore,
    _activate_session_interaction,
    _activate_session_run_fence,
    _checkpoint_with_session_run_operation,
    _deactivate_session_interaction,
    _deactivate_session_run_fence,
    _incomplete_recovery_claim_from_checkpoint,
)
from cayu.sessions.cleanup import (
    RecoveryCleanup,
)
from cayu.sessions.event_queries import EventOrder, EventQuery
from cayu.sessions.interactions import (
    INTERACTION_LIFECYCLE_EVENT_TYPES,
    INTERACTION_TERMINAL_EVENT_TYPES,
)
from cayu.sessions.records import (
    Session,
    SessionStatus,
)
from cayu.tools.exposure import (
    ALL_REGISTERED_TOOLS_PROFILE_ID,
    ResolvedToolExposureAuthority,
)


def _completed_tool_round_model_step(step: int | None, *, max_steps: int) -> int:
    """Retain consumed allowance from the authenticated pending round."""
    if type(step) is not int or not 1 <= step <= max_steps:
        raise RuntimeError("Tool-round continuation has no valid consumed model-step evidence.")
    return step


def _continued_tool_exposure_profile_id(
    authority: ResolvedToolExposureAuthority | None,
) -> str:
    """Return the profile preceding a post-tool continuation model step."""

    if authority is None:
        # Pending rounds written before compact exposure authority was added used
        # the byte-compatible expose-all policy. Execution-profile validation
        # guarantees that the registered catalog and policy did not drift.
        return ALL_REGISTERED_TOOLS_PROFILE_ID
    if type(authority) is not ResolvedToolExposureAuthority:
        raise TypeError("authority must be a ResolvedToolExposureAuthority or None.")
    return authority.profile_id


@dataclass(frozen=True)
class _RecoveryInvocationSemantics:
    """Exact provider-dispatch semantics reconstructed for one continuation."""

    max_steps: int
    limits: RunLimits
    budget_limits: tuple[BudgetLimit, ...]
    retry_policy: RetryPolicy
    structured_output: StructuredOutputSpec | None
    thinking: ThinkingConfig | None


RecoveryMutationHook = Callable[[], Awaitable[None]]

RecoveryExecutionAdmissionHook = Callable[[Session], Awaitable[bool]]


class RecoveryAdmission:
    """Admit a recovered invocation and retain cleanup until handoff settles."""

    def __init__(
        self,
        *,
        clock: Callable[[], datetime],
        environment_lifecycle: EnvironmentLifecycle,
        recovery_ownership: RecoveryOwnership,
        session_control: SessionControl[SessionUsageTracker],
        session_finalization: SessionFinalization,
        session_store: SessionStore,
        terminal_finalization: TerminalEvidenceFinalization,
    ) -> None:
        self._clock = clock
        self._environment_lifecycle = environment_lifecycle
        self._recovery_ownership = recovery_ownership
        self._session_control = session_control
        self._session_finalization = session_finalization
        self._session_store = session_store
        self.terminal_finalization = terminal_finalization

    async def cleanup_recovery_handoff(
        self,
        *,
        stream: AsyncGenerator[Event, None] | None,
        session_id: str,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        authoritative_failure: BaseException | None,
        finalize_abandoned: bool,
        release_run_fence: bool,
        abort_environment_setup: bool = True,
        execution_profile: ExecutionProfileIdentity | None = None,
        invocation_context: InvocationContext | None = None,
    ) -> None:
        if invocation_context is not None:
            # A nested continuation can atomically consume a queued message and
            # transfer the same run epoch from interaction A to interaction B.
            # Recovery entrypoints retain their original local A context, so
            # derive cleanup authority from the durable checkpoint before any
            # owner/fence cleanup observes it.  ``with_queued_interaction``
            # authenticates the complete stable profile and collaborator set;
            # incompatible durable state remains fail-closed below.
            current_session = await self._session_store.load(session_id)
            current_checkpoint = await self._session_store.load_checkpoint(session_id)
            current_profile = active_invocation_execution_profile_from_checkpoint(
                current_checkpoint
            )
            if (
                current_session is not None
                and current_profile is not None
                and current_profile.run_epoch == invocation_context.active_profile.run_epoch
                and current_profile.interaction_id
                != invocation_context.active_profile.interaction_id
            ):
                invocation_context = invocation_context.with_queued_interaction(
                    current_session,
                    active_profile=current_profile,
                )
        cleanup_steps: list[tuple[str, RecoveryCleanup]] = []
        if stream is not None:
            cleanup_steps.append(("nested stream close", stream.aclose))
        if finalize_abandoned:
            cleanup_steps.append(
                (
                    "abandoned session finalization",
                    lambda: self._session_finalization.finalize_abandoned_session_by_id(
                        session_id,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        execution_profile=execution_profile,
                        invocation_context=invocation_context,
                    ),
                )
            )
        if abort_environment_setup and authoritative_failure is not None:
            cleanup_steps.append(
                (
                    "environment setup abort",
                    lambda: self._environment_lifecycle.abort_environment_setup(
                        session_id=session_id,
                        original_error=authoritative_failure,
                        execution_profile=execution_profile,
                        invocation_context=invocation_context,
                    ),
                )
            )
        if release_run_fence:
            cleanup_steps.append(
                (
                    "run fence release",
                    lambda: self._environment_lifecycle.release_run_fence_after_environment_cleanup(
                        session_id=session_id,
                        execution_profile=execution_profile,
                        invocation_context=invocation_context,
                    ),
                )
            )
        await self._recovery_ownership.run_cleanup_steps(
            authoritative_failure=authoritative_failure,
            steps=tuple(cleanup_steps),
        )

    async def cleanup_entrypoint_handoff(
        self,
        *,
        stream: AsyncGenerator[Event, None] | None,
        session_id: str,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        authoritative_failure: BaseException | None,
        finalize_abandoned: bool,
        release_run_fence: bool,
        abort_environment_setup: bool = True,
        execution_profile: ExecutionProfileIdentity | None = None,
        invocation_context: InvocationContext | None = None,
    ) -> None:
        try:
            await self.cleanup_recovery_handoff(
                stream=stream,
                session_id=session_id,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                authoritative_failure=authoritative_failure,
                finalize_abandoned=finalize_abandoned,
                release_run_fence=release_run_fence,
                abort_environment_setup=abort_environment_setup,
                execution_profile=execution_profile,
                invocation_context=invocation_context,
            )
        finally:
            # Cleanup can run in a shielded child task with copied context. The
            # public caller must never retain a stale run epoch after handoff.
            _deactivate_session_run_fence(session_id)
            _deactivate_session_interaction(session_id)

    async def activate_latest_open_interaction(self, session_id: str) -> str | None:
        records = await self._session_store.query_events(
            EventQuery(
                session_id=session_id,
                event_types=INTERACTION_LIFECYCLE_EVENT_TYPES,
                order_by=EventOrder.SEQUENCE_DESC,
                limit=1,
            )
        )
        if not records or records[0].event.type in INTERACTION_TERMINAL_EVENT_TYPES:
            return None
        interaction_id = records[0].event.interaction_id
        if interaction_id is None:
            raise RuntimeError("Interaction lifecycle event has no interaction identity.")
        _activate_session_interaction(session_id, interaction_id)
        return interaction_id

    async def transition_recovery_session_to_running(
        self,
        loaded_session: Session,
        *,
        checkpoint: dict[str, Any] | None,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        from_statuses: set[SessionStatus] | None = None,
        checkpoint_transform: CheckpointTransform | None = None,
        execution_profile_snapshot: ActiveInvocationExecutionProfile | None = None,
        preserve_open_interaction_on_failure: bool = False,
        before_mutation: RecoveryMutationHook | None = None,
        before_resume: RecoveryExecutionAdmissionHook | None = None,
        after_admission: RecoveryMutationHook | None = None,
        invocation_context: InvocationContext | None = None,
    ) -> tuple[Session, Event | None]:
        """Claim a paused session without leaving cancellation outcome-uncertain.

        Session stores activate run fences in task-local context after the durable
        transition commits. Keeping the transition in a shielded child task lets it
        reach a definite result even if the caller is cancelled at that boundary.
        A successful claim is then activated in the caller's context. If cancellation
        arrived, the claim is finalized and released before that cancellation is
        propagated.
        """
        if type(preserve_open_interaction_on_failure) is not bool:
            raise TypeError("preserve_open_interaction_on_failure must be a bool.")
        if invocation_context is not None and (
            invocation_context.binding.session_id != loaded_session.id
            or invocation_context.binding.session_instance_id != loaded_session.instance_id
            or invocation_context.binding.run_epoch != loaded_session.run_epoch
            or registered_agent is not invocation_context.registered_agent
            or registered_environment is not invocation_context.registered_environment
            or execution_profile_snapshot is None
            or invocation_context.profile is not execution_profile_snapshot.profile
        ):
            raise RuntimeError("Recovery admission substituted frozen invocation authority.")
        expected_statuses = (
            {SessionStatus.INTERRUPTED} if from_statuses is None else set(from_statuses)
        )
        if before_mutation is not None:
            preflight_events = await self._session_store.query_events(
                EventQuery(
                    session_id=loaded_session.id,
                    event_types=INTERACTION_LIFECYCLE_EVENT_TYPES,
                    order_by=EventOrder.SEQUENCE_DESC,
                    limit=1,
                )
            )
            if (
                not preflight_events
                or preflight_events[0].event.type in INTERACTION_TERMINAL_EVENT_TYPES
            ):
                raise RuntimeError(
                    "Pending recovery state has no open interaction. "
                    "Pre-interaction prerelease recovery state is unsupported."
                )
            if preflight_events[0].event.interaction_id is None:
                raise RuntimeError("Interaction lifecycle event has no interaction identity.")
            await before_mutation()
        if loaded_session.status in expected_statuses:
            (
                loaded_session,
                checkpoint,
            ) = await self.terminal_finalization.reconcile_before_continuation(
                session=loaded_session,
                checkpoint=checkpoint,
            )
            if execution_profile_snapshot is not None:
                reconciled_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
                if (
                    reconciled_profile is None
                    or reconciled_profile.session_id != execution_profile_snapshot.session_id
                    or reconciled_profile.interaction_id
                    != execution_profile_snapshot.interaction_id
                    or reconciled_profile.profile != execution_profile_snapshot.profile
                ):
                    raise RuntimeError(
                        "Active invocation profile changed during terminal-evidence recovery."
                    )
                execution_profile_snapshot = execution_profile_snapshot.model_copy(
                    update={"run_epoch": reconciled_profile.run_epoch}
                )
        session_id = loaded_session.id
        latest_events = await self._session_store.query_events(
            EventQuery(
                session_id=session_id,
                event_types=INTERACTION_LIFECYCLE_EVENT_TYPES,
                order_by=EventOrder.SEQUENCE_DESC,
                limit=1,
            )
        )
        if not latest_events or latest_events[0].event.type in INTERACTION_TERMINAL_EVENT_TYPES:
            raise RuntimeError(
                "Pending recovery state has no open interaction. "
                "Pre-interaction prerelease recovery state is unsupported."
            )
        interaction_id = latest_events[0].event.interaction_id
        if interaction_id is None:
            raise RuntimeError("Interaction lifecycle event has no interaction identity.")
        if (
            execution_profile_snapshot is not None
            and execution_profile_snapshot.interaction_id != interaction_id
        ):
            raise RuntimeError(
                "Active invocation execution profile belongs to another interaction."
            )
        run_operation_id = str(uuid4())

        def reject_active_incomplete_recovery(
            current_session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any] | None:
            # Expired recovery ownership is reconciled through the store-time
            # claim path above.  This final typed rebind must reject any marker
            # that appeared after that reconciliation instead of interpreting
            # its lease with a worker clock.
            if _incomplete_recovery_claim_from_checkpoint(checkpoint) is not None:
                raise RuntimeError("Session has an active incomplete-session recovery operation.")
            if checkpoint_transform is not None:
                checkpoint = checkpoint_transform(current_session, checkpoint)
            if execution_profile_snapshot is not None:
                current_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
                if current_profile is None or current_profile != execution_profile_snapshot:
                    raise SessionRunFenced(
                        "Active invocation profile changed before continuation claimed it."
                    )
                checkpoint = checkpoint_with_active_invocation_execution_profile(
                    checkpoint,
                    session_id=current_session.id,
                    interaction_id=interaction_id,
                    run_epoch=current_session.run_epoch + 1,
                    profile=execution_profile_snapshot.profile,
                    expected=execution_profile_snapshot,
                )
            return _checkpoint_with_session_run_operation(
                checkpoint=checkpoint,
                current_session=current_session,
                operation_id=run_operation_id,
            )

        transition_task = asyncio.create_task(
            self._recovery_ownership.fence_or_rebind_active_invocation(
                session_id,
                statuses=expected_statuses,
                target_status=SessionStatus.RUNNING,
                checkpoint_transform=reject_active_incomplete_recovery,
            )
        )
        cancellation: asyncio.CancelledError | None = None
        transition_failure: BaseException | None = None
        while not transition_task.done():
            try:
                await asyncio.shield(transition_task)
            except asyncio.CancelledError as exc:
                if transition_task.cancelled():
                    transition_failure = exc
                    break
                if cancellation is None:
                    cancellation = exc
            except BaseException as exc:
                transition_failure = exc
                break

        session: Session | None = None
        if transition_failure is None:
            try:
                session = transition_task.result()
            except BaseException as exc:
                transition_failure = exc

        if session is None:
            if cancellation is not None:
                if transition_failure is not None:
                    cancellation.add_note(
                        "Continuation recovery transition also failed after cancellation: "
                        f"{type(transition_failure).__name__}."
                    )
                raise cancellation from transition_failure
            if transition_failure is None:
                raise RuntimeError("Continuation recovery transition completed without a session.")
            raise transition_failure

        # The transition activated this epoch only in the child task's copied
        # context. The caller owns all subsequent writes and cleanup.
        _activate_session_run_fence(session)
        _activate_session_interaction(session.id, interaction_id)
        rebound_invocation_context = (
            None
            if invocation_context is None or execution_profile_snapshot is None
            else invocation_context.with_rebound_session(
                session,
                active_profile=execution_profile_snapshot.model_copy(
                    update={"run_epoch": session.run_epoch}
                ),
            )
        )
        post_admission_authority_confirmed = after_admission is None
        try:
            if cancellation is not None:
                raise cancellation
            if after_admission is not None:
                await after_admission()
                post_admission_authority_confirmed = True
            # Continuations can execute accepted tools before entering the model
            # loop. Publish this epoch's owner before dispatching any such work.
            await self._session_control.execution_presence.ensure(session)
            if before_resume is not None and not await before_resume(session):
                return session, None
            resumed_event = await self._session_finalization.resume_recovery_interaction(
                session,
                registered_agent,
                registered_environment,
            )
        except BaseException as exc:
            try:
                cleanup_steps: list[tuple[str, RecoveryCleanup]] = []
                if not preserve_open_interaction_on_failure:
                    cleanup_steps.append(
                        (
                            "abandoned session finalization",
                            lambda: self._session_finalization.finalize_abandoned_session_by_id(
                                session.id,
                                registered_agent=registered_agent,
                                registered_environment=registered_environment,
                                execution_profile=(
                                    None
                                    if execution_profile_snapshot is None
                                    else execution_profile_snapshot.profile
                                ),
                                invocation_context=rebound_invocation_context,
                                run_terminal_hooks=post_admission_authority_confirmed,
                            ),
                        )
                    )
                cleanup_steps.append(
                    (
                        "run fence release",
                        lambda: (
                            self._environment_lifecycle.release_run_fence_after_environment_cleanup(
                                session_id=session.id,
                                execution_profile=(
                                    None
                                    if execution_profile_snapshot is None
                                    else execution_profile_snapshot.profile
                                ),
                                invocation_context=rebound_invocation_context,
                            )
                        ),
                    )
                )
                await self._recovery_ownership.run_cleanup_steps(
                    authoritative_failure=exc,
                    steps=tuple(cleanup_steps),
                )
            finally:
                _deactivate_session_run_fence(session.id)
                _deactivate_session_interaction(session.id)
            raise
        if cancellation is None:
            return session, resumed_event

        try:
            cleanup_steps = []
            if not preserve_open_interaction_on_failure:
                cleanup_steps.append(
                    (
                        "abandoned session finalization",
                        lambda: self._session_finalization.finalize_abandoned_session_by_id(
                            session.id,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            execution_profile=(
                                None
                                if execution_profile_snapshot is None
                                else execution_profile_snapshot.profile
                            ),
                            invocation_context=rebound_invocation_context,
                        ),
                    )
                )
            cleanup_steps.append(
                (
                    "run fence release",
                    lambda: self._environment_lifecycle.release_run_fence_after_environment_cleanup(
                        session_id=session.id,
                        execution_profile=(
                            None
                            if execution_profile_snapshot is None
                            else execution_profile_snapshot.profile
                        ),
                        invocation_context=rebound_invocation_context,
                    ),
                )
            )
            await self._recovery_ownership.run_cleanup_steps(
                authoritative_failure=cancellation,
                steps=tuple(cleanup_steps),
            )
        finally:
            # Shielded cleanup runs in a copied context. Never leave the caller's
            # task-local epoch active if it catches and handles the cancellation.
            _deactivate_session_run_fence(session.id)
            _deactivate_session_interaction(session.id)
        raise cancellation

    async def prepare_recovered_tool_round_continuation(
        self,
        *,
        session: Session,
        pending_round: pending_rounds.PendingToolRound,
        invocation_semantics: _RecoveryInvocationSemantics,
        invocation_context: InvocationContext,
        request_metadata: dict[str, Any],
        task_worker_id: str | None,
        task_handoff_id: str | None,
        completed_model_step: int | None = None,
        participant_context: CollaborationAccessContext | None = None,
    ) -> RecoverySessionRunRequest:
        """Restore a claimed round without deciding its outcome or acquiring a new owner.

        Receipt settlement and operator recovery share continuation preparation.
        The caller retains the recovery claim, stream supervision, and cleanup.
        """
        if type(invocation_context) is not InvocationContext:
            raise TypeError("Recovered continuation requires authenticated invocation authority.")
        invocation_context._validate()
        invocation_context.with_admitted_session(session)
        if pending_round.agent_name != invocation_context.registered_agent.spec.name:
            raise RuntimeError("Recovered continuation belongs to another agent.")
        transcript = await self._session_store.load_transcript(session.id)
        continued_run_limit_accounting = pending_round.run_limit_accounting
        if continued_run_limit_accounting is not None:
            recovery_events = await self._session_store.load_events(session.id)
            continued_run_limit_accounting = rebase_run_limit_accounting_context(
                continued_run_limit_accounting,
                session_id=session.id,
                limits=invocation_semantics.limits,
                budget_limits=request_budget_limits_for_session(
                    limits=invocation_semantics.budget_limits,
                    agent_name=invocation_context.registered_agent.spec.name,
                    causal_budget_id=session.causal_budget_id,
                ),
                events=recovery_events,
                reset_run_limits=False,
                reset_budgets=False,
                now=self._clock(),
            )
        return RecoverySessionRunRequest(
            session=session,
            participant_context=participant_context,
            invocation_context=invocation_context,
            messages=transcript,
            messages_to_append=[],
            max_steps=invocation_semantics.max_steps,
            limits=invocation_semantics.limits,
            budget_limits=invocation_semantics.budget_limits,
            retry_policy=invocation_semantics.retry_policy,
            structured_output=invocation_semantics.structured_output,
            thinking=invocation_semantics.thinking,
            request_metadata=copy_durable_metadata(request_metadata, "recovery.request_metadata"),
            task_id=pending_round.task_id,
            task_worker_id=task_worker_id,
            task_handoff_id=task_handoff_id,
            start_event_type=None,
            start_event_payload={},
            start_task_on_enter=False,
            release_run_fence_on_exit=False,
            run_limit_accounting=continued_run_limit_accounting,
            completed_tool_round_model_step=_completed_tool_round_model_step(
                pending_round.model_step if completed_model_step is None else completed_model_step,
                max_steps=invocation_semantics.max_steps,
            ),
            previous_tool_exposure_profile_id=(
                _continued_tool_exposure_profile_id(pending_round.tool_exposure)
            ),
        )
