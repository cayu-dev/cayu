"""Own manual model settlement, terminal election, replay and claim cleanup."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any

from cayu._validation import canonical_durable_json_bytes, require_clean_nonblank
from cayu.budgets.base import BudgetPolicy
from cayu.events import Event, EventType, event_with_runtime_envelope_authority
from cayu.execution_units import ModelAttemptIdentity
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _session_execution_profile
from cayu.runtime._durable_model_terminalization import (
    inspect_terminalization,
    require_terminalization_checkpoint,
    terminalization_plan_owner,
)
from cayu.runtime._environment_lifecycle import EnvironmentLifecycle
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime._execution_profile_continuation import ExecutionProfileContinuation
from cayu.runtime._invocation_lifecycle import _authenticated_invocation_context
from cayu.runtime._model_completion_contracts import (
    ModelCompletionManualRecoveryRequired,
    model_completion_recovery_context_from_stage,
)
from cayu.runtime._recovery_claims import _IncompleteRecoveryClaim
from cayu.runtime._recovery_ownership import RecoveryOwnership
from cayu.runtime._run_limits import RunLimitController
from cayu.runtime.loop_policies import LoopPolicy
from cayu.sessions._execution_profile_checkpoint import (
    active_invocation_execution_profile_from_checkpoint,
)
from cayu.sessions._interaction_transitions import (
    InteractionTransitionResult,
    InteractionTransitionSpec,
)
from cayu.sessions._invocation_lifecycle import AdmittedInvocationBinding, SettleInvocationCommand
from cayu.sessions._invocation_terminal_decision import (
    InvocationTerminalDecision,
    InvocationTerminalOutcome,
    build_invocation_terminal_decision,
    checkpoint_with_invocation_terminal_decision,
    invocation_terminal_decision_from_checkpoint,
    invocation_terminal_event_id,
    settled_invocation_terminal_decision_from_checkpoint,
)
from cayu.sessions._model_completion_publication import (
    ModelCompletionStageDisposition,
    ModelCompletionStageSettlementRequest,
)
from cayu.sessions._model_stage import (
    ModelCompletionManualRecoveryRequest,
    ModelCompletionManualRecoveryResult,
    ModelCompletionStageSettlement,
    SessionModelCompletionStageConflict,
    copy_model_completion_manual_recovery_request,
)
from cayu.sessions._model_stage_rules import (
    model_completion_stage_settlement_request,
    require_terminalization_plan_owner,
)
from cayu.sessions.authority import SessionRunFenced
from cayu.sessions.records import Session, SessionStatus, SessionStatusConflict
from cayu.sessions.recovery import _incomplete_recovery_claim_from_checkpoint
from cayu.sessions.store import SessionStore

_TERMINALIZATION_DEADLINE_SECONDS = 30


class ManualModelCompletionRecovery:
    """Settle ambiguous dispatched model work without executing a session turn.

    Registration resolvers authenticate ordinary recovery. Terminalization-only
    recovery instead uses the persisted profile and the recovery claim owner.
    """

    def __init__(
        self,
        *,
        session_store: SessionStore,
        event_writer: RuntimeEventWriter,
        run_limit_controller: RunLimitController,
        get_registered_agent: Callable[[str], runtime_records.RegisteredAgentState],
        get_registered_provider: Callable[[str], runtime_records.RegisteredProvider],
        get_registered_environment_for_session: Callable[
            [str | None], runtime_records.RegisteredEnvironment | None
        ],
        get_budget_policy: Callable[[], BudgetPolicy | None],
        execution_profile_continuation: ExecutionProfileContinuation,
        runtime_hooks: tuple[runtime_records.RegisteredRuntimeHook, ...],
        loop_policies: tuple[LoopPolicy, ...],
        environment_lifecycle: EnvironmentLifecycle,
        recovery_ownership: RecoveryOwnership,
    ) -> None:
        self.session_store = session_store
        self._event_writer = event_writer
        self._run_limit_controller = run_limit_controller
        self._get_registered_agent = get_registered_agent
        self._get_registered_provider = get_registered_provider
        self._get_registered_environment_for_session = get_registered_environment_for_session
        self._get_budget_policy = get_budget_policy
        self._execution_profile_continuation = execution_profile_continuation
        self._runtime_hooks = runtime_hooks
        self._loop_policies = loop_policies
        self._environment_lifecycle = environment_lifecycle
        self._recovery_ownership = recovery_ownership

    async def recover_model_completion_stage(
        self,
        request: ModelCompletionManualRecoveryRequest,
    ) -> ModelCompletionManualRecoveryResult:
        """Conservatively settle one ambiguous model dispatch before clearing it."""

        request = copy_model_completion_manual_recovery_request(request)
        if request.terminalization_only:
            return await self._terminalize_dispatched_model(request)
        return await self._recover_model_completion_stage(request)

    async def _recover_model_completion_stage(
        self,
        request: ModelCompletionManualRecoveryRequest,
        *,
        terminalization: tuple[_IncompleteRecoveryClaim, InvocationTerminalDecision, Event]
        | None = None,
    ) -> ModelCompletionManualRecoveryResult:
        from cayu.runtime._durable_model_terminalization import terminalization_plan_owner

        request = copy_model_completion_manual_recovery_request(request)
        session = await self.session_store.load(request.session_id)
        if session is None:
            raise KeyError(f"Session not found: {request.session_id}")
        stage = await self.session_store.load_model_completion_stage(
            request.session_id,
            request.stage_id,
        )
        if stage is None or stage.state != "in_flight":
            raise SessionModelCompletionStageConflict(
                "Manual model recovery requires one exact in-flight stage."
            )
        recovery_context = model_completion_recovery_context_from_stage(stage)
        if recovery_context is None:
            raise SessionModelCompletionStageConflict(
                "Manual model recovery requires durable accounting and profile authority."
            )
        interaction_id = stage.intent.get("interaction_id")
        provider_name = stage.intent.get("provider_name")
        pricing_provider_name = stage.intent.get("pricing_provider_name", provider_name)
        model_attempt_id = stage.intent.get("model_attempt_id")
        if not all(
            type(value) is str
            for value in (
                interaction_id,
                provider_name,
                pricing_provider_name,
                model_attempt_id,
            )
        ):
            raise SessionModelCompletionStageConflict(
                "Manual model recovery lost its interaction, provider, or attempt identity."
            )
        assert isinstance(interaction_id, str)
        assert isinstance(provider_name, str)
        assert isinstance(pricing_provider_name, str)
        assert isinstance(model_attempt_id, str)
        from cayu.runtime._model_failover_stage import model_failover_target_for_stored_stage

        selected_target = model_failover_target_for_stored_stage(session=session, stage=stage)
        if (
            stage.purpose == "assistant-turn"
            and selected_target is None
            and provider_name != session.provider_name
        ):
            raise SessionModelCompletionStageConflict(
                "Assistant model recovery provider conflicts with the session provider."
            )
        model_attempt_identity = ModelAttemptIdentity(
            model_step_id=stage.logical_step_id,
            model_attempt_id=model_attempt_id,
        )
        budget_dispatch_id = stage.stage_id
        budget_model = session.model if selected_target is None else selected_target.model
        if stage.purpose == "context-compaction":
            requested_model = stage.intent.get("requested_model")
            if type(requested_model) is not str:
                raise SessionModelCompletionStageConflict(
                    "Context-compaction recovery lost its requested model."
                )
            try:
                budget_model = require_clean_nonblank(requested_model, "requested_model")
            except ValueError as invalid_model:
                raise SessionModelCompletionStageConflict(
                    "Context-compaction recovery has an invalid requested model."
                ) from invalid_model
            budget_dispatch_id = model_attempt_id

        expected_settlement = model_completion_stage_settlement_request(
            stage,
            interaction_id=interaction_id,
            disposition=ModelCompletionStageDisposition.PROVIDER_EFFECT_OUTCOME_UNKNOWN,
            reason_code="operator_outcome_unknown",
            execution_profile_fingerprint=(recovery_context.execution_profile_fingerprint),
            settlement_run_epoch=request.expected_run_epoch,
            settled_reservation_ids=stage.reservation_ids,
        )

        def validate_replay(
            settlement: ModelCompletionStageSettlement,
        ) -> ModelCompletionStageSettlement:
            replay_request = ModelCompletionStageSettlementRequest.model_validate(
                settlement.model_dump(
                    mode="python",
                    include=set(ModelCompletionStageSettlementRequest.model_fields),
                )
            )
            if replay_request != expected_settlement:
                raise SessionModelCompletionStageConflict(
                    "Manual model recovery conflicts with the durable stage settlement."
                )
            return settlement

        prior = await self.session_store.load_model_completion_stage_settlement(
            request.session_id,
            request.stage_id,
        )
        checkpoint = await self.session_store.load_checkpoint(request.session_id)
        if terminalization is not None:
            from cayu.runtime._durable_model_terminalization import (
                require_terminalization_checkpoint,
            )

            active_profile = require_terminalization_checkpoint(session, checkpoint)
            if (
                active_profile.interaction_id != interaction_id
                or active_profile.profile.fingerprint
                != recovery_context.execution_profile_fingerprint
            ):
                raise SessionModelCompletionStageConflict(
                    "Model terminalization lost exact stage/profile authority."
                )
        else:
            try:
                registered_agent = self._get_registered_agent(session.agent_name)
                # A context-compaction stage may belong to a provider other than
                # the session's assistant provider. Invocation-profile authority
                # remains the session provider; the staged effect provider below
                # is used only for exact budget settlement.
                registered_provider = self._get_registered_provider(session.provider_name)
                registered_environment = self._get_registered_environment_for_session(
                    session.environment_name
                )
            except (KeyError, RuntimeError) as registration_error:
                raise ModelCompletionManualRecoveryRequired(
                    "Manual model recovery requires the original agent, provider, and "
                    "environment registrations."
                ) from registration_error
            budget_policy = self._get_budget_policy()
            active_profile = await self._execution_profile_continuation.validate(
                session=session,
                checkpoint=checkpoint,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                budget_policy=budget_policy,
                require_open_interaction=prior is None,
            )
            _authenticated_invocation_context(
                active_profile=active_profile,
                binding=AdmittedInvocationBinding(
                    session_id=session.id,
                    session_instance_id=session.instance_id,
                    interaction_id=active_profile.interaction_id,
                    run_epoch=session.run_epoch,
                    agent_name=session.agent_name,
                    provider_name=session.provider_name,
                    model=session.model,
                    runtime_name=session.runtime_name,
                    runtime_version=session.runtime_version,
                    runtime_build_provenance=session.runtime_build_provenance,
                    environment_name=session.environment_name,
                ),
                validated_profile=active_profile.profile,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                registered_environment=registered_environment,
                runtime_hooks=self._runtime_hooks,
                loop_policies=self._loop_policies,
                request_loop_policies=(),
                budget_policy=budget_policy,
                tool_capability_ceiling=_session_execution_profile._session_tool_capability_ceiling(
                    session
                ),
            )
        if prior is not None:
            budget_events = await self._run_limit_controller.recover_pending_budget_settlements(
                session_id=request.session_id
            )
            borrowed_budget_events = await (
                self._run_limit_controller.reconcile_borrowed_automatic_compaction_budget_authority(
                    session=session,
                    stage=stage,
                    allow_outcome_unknown=True,
                )
            )
            budget_events = list(
                {event.id: event for event in (*budget_events, *borrowed_budget_events)}.values()
            )
            await self._run_limit_controller.require_model_completion_reservation_settlements(
                reservation_ids=stage.reservation_ids,
                recovery_contexts=recovery_context.budget_reservations,
                dispatch_id=budget_dispatch_id,
            )
            current = await self.session_store.load(request.session_id)
            if current is None:
                raise KeyError(f"Session not found: {request.session_id}")
            if current.status is not request.terminal_status:
                raise SessionStatusConflict(
                    "Manual model recovery replay conflicts with the terminal session status."
                )
            return ModelCompletionManualRecoveryResult(
                session=current,
                settlement=validate_replay(prior),
                budget_events=tuple(budget_events),
                replayed=True,
            )

        if session.run_epoch != request.expected_run_epoch:
            raise SessionRunFenced(
                "Manual model recovery run epoch is stale: expected "
                f"{request.expected_run_epoch}, current {session.run_epoch}."
            )
        if session.status not in {
            SessionStatus.RUNNING,
            SessionStatus.INTERRUPTING,
            request.terminal_status,
        }:
            raise SessionStatusConflict(
                "Manual model recovery cannot replace a different terminal session outcome."
            )
        active = await self.session_store.load_active_model_completion_stage(request.session_id)
        if active is None or active.stage != stage:
            raise SessionModelCompletionStageConflict(
                "Manual model recovery no longer owns the exact active stage."
            )
        if stage.intent.get("provider_operation_start") is not None:
            raise ModelCompletionManualRecoveryRequired(
                "Background provider work must use provider-operation resolution."
            )
        dispatch = await self.session_store.load_model_completion_stage_dispatch(
            request.session_id,
            request.stage_id,
        )
        if dispatch is None:
            raise SessionModelCompletionStageConflict(
                "Receipt-less model completion must use automatic pre-provider recovery."
            )

        borrowed_budget_events = await (
            self._run_limit_controller.reconcile_borrowed_automatic_compaction_budget_authority(
                session=session,
                stage=stage,
                allow_outcome_unknown=True,
            )
        )
        stage_budget_events = (
            await self._run_limit_controller.reconcile_manual_model_completion_reservations(
                reservation_ids=stage.reservation_ids,
                recovery_contexts=recovery_context.budget_reservations,
                session=session,
                provider_name=pricing_provider_name,
                model=budget_model,
                model_attempt_identity=model_attempt_identity,
                dispatch_id=budget_dispatch_id,
                is_automatic_context_compaction=(stage.purpose == "context-compaction"),
            )
        )
        budget_events = list(
            {event.id: event for event in (*borrowed_budget_events, *stage_budget_events)}.values()
        )
        await self._run_limit_controller.require_model_completion_reservation_settlements(
            reservation_ids=stage.reservation_ids,
            recovery_contexts=recovery_context.budget_reservations,
            dispatch_id=budget_dispatch_id,
        )

        current = await self.session_store.load(request.session_id)
        current_active = await self.session_store.load_active_model_completion_stage(
            request.session_id
        )
        if current is None:
            raise KeyError(f"Session not found: {request.session_id}")
        if current.run_epoch != request.expected_run_epoch:
            raise SessionRunFenced(
                "Manual model recovery lost its exact run epoch after budget settlement."
            )
        if current_active is None or current_active.stage != stage:
            raise SessionModelCompletionStageConflict(
                "Manual model recovery lost its active stage after budget settlement."
            )

        recovery_event = event_with_runtime_envelope_authority(
            Event(
                id="evt_model_completion_manual_"
                + hashlib.sha256(
                    canonical_durable_json_bytes(
                        request.model_dump(mode="json"),
                        "model_completion_manual_recovery",
                    )
                ).hexdigest(),
                type=(
                    EventType.INTERACTION_FAILED
                    if request.terminal_status is SessionStatus.FAILED
                    else EventType.INTERACTION_INTERRUPTED
                ),
                session_id=request.session_id,
                interaction_id=interaction_id,
                agent_name=current.agent_name,
                environment_name=current.environment_name,
            ),
            "session_id",
            "interaction_id",
        )
        if terminalization is not None:
            decision = terminalization[1]
            recovery_event = recovery_event.model_copy(
                update={
                    "id": decision.interaction_event_id,
                    "timestamp": decision.observed_at,
                }
            )
        recovery_event = self._event_writer.prepare(recovery_event)
        try:
            transition_result = await self.session_store.apply_invocation_lifecycle_command(
                SettleInvocationCommand(
                    session_id=request.session_id,
                    expected_session_instance_id=current.instance_id,
                    expected_run_epoch=current.run_epoch,
                    expected_active_profile=active_profile,
                    recovery_claim_id=None
                    if terminalization is None
                    else terminalization[0].claim_id,
                    terminalization_only=terminalization is not None,
                    terminalization_plan_ownership=terminalization_plan_owner()
                    if terminalization is not None
                    else None,
                    transition=InteractionTransitionSpec(
                        event=recovery_event,
                        from_statuses=(current.status,),
                        to_status=request.terminal_status,
                        model_completion_stage_settlement=expected_settlement,
                        terminal_event=None if terminalization is None else terminalization[2],
                        terminal_decision=None if terminalization is None else terminalization[1],
                    ),
                )
            )
            if type(transition_result) is not InteractionTransitionResult:
                raise RuntimeError("Manual recovery settlement returned an invalid result.")
            transition = transition_result
        except Exception:
            reconciled = await self.session_store.load_model_completion_stage_settlement(
                request.session_id,
                request.stage_id,
            )
            if reconciled is None:
                raise
            validate_replay(reconciled)
            transitioned = await self.session_store.load(request.session_id)
            if transitioned is None:
                raise KeyError(f"Session not found: {request.session_id}") from None
            if transitioned.status is not request.terminal_status:
                raise SessionStatusConflict(
                    "Manual model recovery replay conflicts with the terminal session status."
                ) from None
            return ModelCompletionManualRecoveryResult(
                session=transitioned,
                settlement=reconciled,
                budget_events=tuple(budget_events),
                replayed=True,
            )

        await self._event_writer.fan_out_persisted(
            [transition.event]
            + ([] if transition.terminal_event is None else [transition.terminal_event])
        )
        settlement = await self.session_store.load_model_completion_stage_settlement(
            request.session_id,
            request.stage_id,
        )
        if settlement is None:
            raise RuntimeError("Manual model recovery committed without stage settlement.")
        if transition.session.status is not request.terminal_status:
            raise SessionStatusConflict(
                "Manual model recovery committed an unexpected terminal session status."
            )
        return ModelCompletionManualRecoveryResult(
            session=transition.session,
            settlement=validate_replay(settlement),
            budget_events=tuple(budget_events),
            replayed=transition.replayed,
        )

    async def _terminalize_dispatched_model(
        self, request: ModelCompletionManualRecoveryRequest
    ) -> ModelCompletionManualRecoveryResult:
        store = self.session_store
        plan_owner = terminalization_plan_owner()
        session = await store.load(request.session_id)
        if session is None:
            raise KeyError("Session not found.")
        if session.instance_id != request.expected_session_instance_id:
            raise SessionRunFenced(
                "Model terminalization request belongs to another session incarnation."
            )
        checkpoint = await store.load_checkpoint(session.id)
        # Claim epochs and inactivity bounds describe admission of an attempt, not
        # the operator's disposition of this exact stage. Replanning after a failed
        # publication must retain the elected decision across a new recovery epoch.
        request_digest = sha256(
            canonical_durable_json_bytes(
                request.model_dump(
                    mode="json", exclude={"expected_run_epoch", "inactive_for_seconds"}
                ),
                "model terminalization",
            )
        ).hexdigest()
        source_id = "model-recovery:" + request_digest
        elected = invocation_terminal_decision_from_checkpoint(checkpoint)
        settled_decision = settled_invocation_terminal_decision_from_checkpoint(checkpoint)
        previous = elected or settled_decision
        matching_decision = previous is not None and (
            (previous.model_recovery_id or previous.interruption_request_id) == source_id
            and previous.session_instance_id == session.instance_id
            and previous.session_id == session.id
        )
        if elected is not None and not matching_decision:
            raise SessionRunFenced("Another model terminalization decision owns this invocation.")
        if request.expected_run_epoch > session.run_epoch or (
            not matching_decision and session.run_epoch != request.expected_run_epoch
        ):
            raise SessionRunFenced("Model terminalization request has a stale run epoch.")
        prior = await store.load_model_completion_stage_settlement(session.id, request.stage_id)
        if prior is not None:
            if (
                not matching_decision
                or session.status is not request.terminal_status
                or session.run_epoch
                not in {prior.settlement_run_epoch, prior.settlement_run_epoch + 1}
            ):
                raise SessionRunFenced("Model terminalization replay conflicts with durable state.")
            # Paired terminal evidence and exact release are authenticated by the
            # existing invocation cleanup boundary, including commit-before-ack.
            from cayu.sessions._invocation_lifecycle import (
                require_released_invocation_command_authority,
            )
            from cayu.sessions._invocation_scope import _activate_owned_session_run_fence

            if session.run_epoch == prior.settlement_run_epoch:
                owner = _activate_owned_session_run_fence(session)
                try:
                    await self._environment_lifecycle.release_run_fence_after_environment_cleanup(
                        session_id=session.id
                    )
                finally:
                    owner.retire()
                session = await store.load(request.session_id)
                if session is None:
                    raise SessionRunFenced(
                        "Model terminalization session disappeared during release."
                    )
                checkpoint = await store.load_checkpoint(session.id)
            profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
            if profile is None:
                raise SessionRunFenced("Model terminalization replay lost its released profile.")
            require_released_invocation_command_authority(
                session,
                checkpoint,
                session_id=session.id,
                session_instance_id=session.instance_id,
                active_profile=profile,
            )
            current = await store.load(session.id)
            if current is None:
                raise SessionRunFenced("Model terminalization session disappeared during replay.")
            return ModelCompletionManualRecoveryResult(
                session=current, settlement=prior, replayed=True
            )
        try:
            active = await inspect_terminalization(store, session, checkpoint)
        except ModelCompletionManualRecoveryRequired as error:
            # These preflight reads are not a transaction. Another terminalizer
            # can settle after our settlement lookup and before stage inspection.
            # Positive owner advancement fences this attempt; it is not evidence
            # that an unchanged invocation requires a different recovery policy.
            current = await store.load(session.id)
            if current is None or (
                current.instance_id != session.instance_id
                or current.run_epoch != session.run_epoch
                or current.status != session.status
            ):
                raise SessionRunFenced(
                    "Model terminalization owner changed during preflight; reconcile exact state."
                ) from error
            raise
        if active.stage.stage_id != request.stage_id:
            raise SessionRunFenced("Model terminalization selected another active stage.")
        source_profile = require_terminalization_checkpoint(session, checkpoint)
        decision: InvocationTerminalDecision | None = None

        def claim_transform(current: Session, updated: dict[str, Any] | None) -> dict[str, Any]:
            if updated is None:
                raise SessionRunFenced("Model terminalization claim has no checkpoint.")
            nonlocal decision
            # The transform runs under the same transaction that advances the epoch.
            profile = active_invocation_execution_profile_from_checkpoint(updated)
            if profile is None:
                raise SessionRunFenced("Model terminalization claim lost its active profile.")
            require_terminalization_checkpoint(
                current.model_copy(update={"run_epoch": profile.run_epoch}), updated
            )
            require_terminalization_plan_owner(
                updated,
                plan_owner,
                datetime.fromisoformat(updated["incomplete_session_recovery_claim"]["claimed_at"]),
            )
            existing = invocation_terminal_decision_from_checkpoint(updated)
            if existing is not None:
                if (existing.model_recovery_id or existing.interruption_request_id) != source_id:
                    raise SessionRunFenced("Another terminal outcome has already been elected.")
                decision = existing
                return updated
            outcome = InvocationTerminalOutcome(request.terminal_status.value)
            observed = datetime.now(UTC)
            if outcome is InvocationTerminalOutcome.INTERRUPTED:
                event_ids = [
                    invocation_terminal_event_id(
                        outcome=outcome,
                        session_id=current.id,
                        session_instance_id=current.instance_id,
                        run_epoch=profile.run_epoch,
                        interaction_id=profile.interaction_id,
                        source_id=source_id,
                        event_kind=kind,
                    )
                    for kind in ("interaction", "session")
                ]
            else:
                event_ids = [source_id + ":interaction_failed", source_id + ":session_failed"]
            decision = build_invocation_terminal_decision(
                outcome=outcome,
                session_id=current.id,
                session_instance_id=current.instance_id,
                run_epoch=profile.run_epoch,
                profile_interaction_id=profile.interaction_id,
                interaction_id=profile.interaction_id,
                execution_profile_fingerprint=profile.profile.fingerprint,
                interaction_event_id=event_ids[0],
                terminal_event_id=event_ids[1],
                observed_at=observed,
                terminal_payload={
                    "reason": "operator_model_outcome_unknown",
                    "terminalization_only": True,
                },
                interruption_request_id=source_id
                if outcome is InvocationTerminalOutcome.INTERRUPTED
                else None,
                model_recovery_id=source_id
                if outcome is InvocationTerminalOutcome.FAILED
                else None,
            )
            return updated

        ownership = self._recovery_ownership
        claim = await ownership.claim(
            session=session,
            inactive_for_seconds=request.inactive_for_seconds,
            execution_profile_snapshot=source_profile,
            checkpoint_transform=claim_transform,
        )
        if claim is None:
            raise SessionRunFenced(
                "Model terminalization could not acquire an inactive recovery owner."
            )

        async def recover_owned() -> ModelCompletionManualRecoveryResult:
            current = await store.load(session.id)
            if current is None:
                raise SessionRunFenced("Model terminalization session disappeared.")
            confirmed = await inspect_terminalization(
                store, current, await store.load_checkpoint(session.id)
            )
            if confirmed != active:
                raise SessionRunFenced("Active model stage changed during terminalization claim.")
            assert decision is not None
            from cayu.sessions._checkpoint_preservation import (
                _invocation_lifecycle_authority_mutation_scope,
            )

            def elect(
                current_session: Session, current_checkpoint: dict[str, Any] | None, now: datetime
            ) -> dict[str, Any]:
                assert decision is not None
                owned = _incomplete_recovery_claim_from_checkpoint(current_checkpoint)
                if (
                    current_session.instance_id != session.instance_id
                    or current_session.run_epoch != claim.session.run_epoch
                    or owned is None
                    or owned[0] != claim.claim_id
                    or owned[1] <= now
                ):
                    raise SessionRunFenced("Model terminalization lost its claim before election.")
                require_terminalization_plan_owner(current_checkpoint, plan_owner, now)
                require_terminalization_checkpoint(current_session, current_checkpoint)
                return checkpoint_with_invocation_terminal_decision(current_checkpoint, decision)

            with _invocation_lifecycle_authority_mutation_scope():
                await store.transform_checkpoint_with_store_time(session.id, elect)
            terminal = self._event_writer.prepare(
                event_with_runtime_envelope_authority(
                    Event(
                        id=decision.terminal_event_id,
                        type=EventType.SESSION_FAILED
                        if request.terminal_status is SessionStatus.FAILED
                        else EventType.SESSION_INTERRUPTED,
                        session_id=session.id,
                        agent_name=session.agent_name,
                        timestamp=decision.observed_at,
                        payload=decision.terminal_payload,
                    ),
                    "session_id",
                )
            )
            return await self._recover_model_completion_stage(
                request.model_copy(update={"expected_run_epoch": claim.session.run_epoch}),
                terminalization=(claim, decision, terminal),
            )

        failure = None
        try:
            async with asyncio.timeout(_TERMINALIZATION_DEADLINE_SECONDS):
                result = await ownership.run_with_heartbeat(
                    claim=claim,
                    recovery=recover_owned,
                )
        except BaseException as exc:
            failure = exc
            raise
        finally:
            await ownership.cleanup_claim(
                authority=claim.require_authority(),
                authoritative_failure=failure,
            )
        current = await store.load(session.id)
        if current is None:
            raise SessionRunFenced("Model terminalization session disappeared after settlement.")
        return result.model_copy(update={"session": current})
