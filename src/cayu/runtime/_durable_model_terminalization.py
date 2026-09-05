"""Terminal-only model recovery under the existing incomplete-recovery claim.

Executable registrations never enter this path. Unknown dependent work blocks it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from hashlib import sha256
from typing import TYPE_CHECKING, Any

from cayu._validation import canonical_durable_json_bytes
from cayu.core.events import Event, EventType, event_with_runtime_envelope_authority
from cayu.runtime._durable_operation_ownership import DurableOperationOwnership
from cayu.runtime._invocation_terminal_decision import (
    InvocationTerminalDecision,
    InvocationTerminalOutcome,
    build_invocation_terminal_decision,
    checkpoint_with_invocation_terminal_decision,
    invocation_terminal_decision_from_checkpoint,
    invocation_terminal_event_id,
    settled_invocation_terminal_decision_from_checkpoint,
)
from cayu.runtime._model_step_executor import model_completion_recovery_context_from_stage
from cayu.runtime._recovery_coordinator import ModelCompletionManualRecoveryRequired
from cayu.runtime.execution_profiles import (
    ActiveInvocationExecutionProfile,
    active_invocation_execution_profile_from_checkpoint,
)
from cayu.runtime.sessions import (
    ActiveModelCompletionStage,
    ModelCompletionManualRecoveryRequest,
    ModelCompletionManualRecoveryResult,
    Session,
    SessionExecutionSource,
    SessionQuery,
    SessionRunFenced,
    SessionStatus,
    SessionStore,
    _validate_model_completion_stage_dispatch,
)

if TYPE_CHECKING:
    from cayu.runtime._session_engine import SessionEngine

_PLAN_OWNER: ContextVar[DurableOperationOwnership | None] = ContextVar(
    "model_terminalization_plan_owner", default=None
)


@contextmanager
def terminalization_plan_scope(ownership: DurableOperationOwnership) -> Iterator[None]:
    token = _PLAN_OWNER.set(ownership)
    try:
        yield
    finally:
        _PLAN_OWNER.reset(token)


def terminalization_plan_owner() -> DurableOperationOwnership | None:
    return _PLAN_OWNER.get()


def require_terminalization_plan_owner(
    checkpoint: dict[str, Any] | None,
    expected: DurableOperationOwnership | None,
    now: datetime,
) -> None:
    marker = None if checkpoint is None else checkpoint.get("recovery_plan_execution")
    if marker is None:
        if expected is not None:
            raise SessionRunFenced("Model terminalization lost its plan owner.")
        return
    if type(marker) is not dict or expected is None:
        raise SessionRunFenced("Model terminalization is owned by another recovery plan.")
    current = DurableOperationOwnership.model_validate(marker.get("ownership"))
    if (
        current.claim_id != expected.claim_id
        or current.generation != expected.generation
        or current.owner_id != expected.owner_id
        or current.operation_id != expected.operation_id
        or current.state.value != "active"
        or current.lease_expires_at is None
        or current.lease_expires_at <= now
    ):
        raise SessionRunFenced("Model terminalization recovery plan expired or changed.")


_TERMINALIZATION_DEADLINE_SECONDS = 30
_ALLOWED_KEYS = {
    "checkpoint_schema_version",
    "active_invocation_execution_profile",
    "invocation_lifecycle_receipt",
    "invocation_terminal_decision",
    "settled_invocation_terminal_decision",
    "incomplete_session_recovery_claim",
    "recovery_plan_execution",
}


def require_terminalization_checkpoint(
    session: Session, checkpoint: dict[str, Any] | None
) -> ActiveInvocationExecutionProfile:
    if (
        session.environment_name is not None
        or session.parent_session_id is not None
        or session.invocation.source
        not in {SessionExecutionSource.SDK_RUN, SessionExecutionSource.HTTP_RUN}
        or not checkpoint
        or set(checkpoint) - _ALLOWED_KEYS
    ):
        raise ModelCompletionManualRecoveryRequired(
            "Terminalization-only recovery requires a standalone model invocation without dependent work."
        )
    active = active_invocation_execution_profile_from_checkpoint(checkpoint)
    if active is None or active.session_id != session.id or active.run_epoch != session.run_epoch:
        raise SessionRunFenced("Terminalization-only recovery lost its durable active profile.")
    return active


async def inspect_terminalization(
    store: SessionStore, session: Session, checkpoint: dict[str, Any] | None
) -> ActiveModelCompletionStage:
    if getattr(store, "durable_model_terminalization_version", None) != 1:
        raise ModelCompletionManualRecoveryRequired(
            "Store does not support atomic model terminalization."
        )
    profile = require_terminalization_checkpoint(session, checkpoint)
    children = await store.list_sessions(SessionQuery(parent_session_id=session.id, limit=1))
    if children.sessions:
        raise ModelCompletionManualRecoveryRequired("Model terminalization has dependent children.")
    active = await store.load_active_model_completion_stage(session.id)
    if active is None or active.stage.state != "in_flight":
        raise ModelCompletionManualRecoveryRequired(
            "Model terminalization requires an in-flight stage."
        )
    stage = active.stage
    context = model_completion_recovery_context_from_stage(stage)
    # Automatic compaction can carry borrowed accounting and pending context
    # publication; it remains on exact-profile recovery until separately proven.
    if (
        stage.purpose != "assistant-turn"
        or stage.intent.get("provider_operation_start") is not None
    ):
        raise ModelCompletionManualRecoveryRequired(
            "Compaction and background provider stages require ordinary recovery."
        )
    if (
        context is None
        or context.task_id is not None
        or stage.session_id != session.id
        or stage.source_run_epoch > session.run_epoch
        or context.interaction_id != stage.intent.get("interaction_id")
        or context.execution_profile_fingerprint != profile.profile.fingerprint
        or context.interaction_id != profile.interaction_id
        or tuple(r.reservation_id for r in context.budget_reservations) != stage.reservation_ids
        or stage.intent.get("provider_name") != session.provider_name
    ):
        raise ModelCompletionManualRecoveryRequired(
            "Model terminalization lacks exact durable accounting/profile authority."
        )
    dispatch = await store.load_model_completion_stage_dispatch(session.id, stage.stage_id)
    if dispatch is None:
        raise ModelCompletionManualRecoveryRequired(
            "Receipt-less stages require pre-provider recovery."
        )
    _validate_model_completion_stage_dispatch(dispatch, stage)
    return active


async def terminalize_dispatched_model(
    engine: SessionEngine, request: ModelCompletionManualRecoveryRequest
) -> ModelCompletionManualRecoveryResult:
    store = engine.session_store
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
            request.model_dump(mode="json", exclude={"expected_run_epoch", "inactive_for_seconds"}),
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
            or session.run_epoch not in {prior.settlement_run_epoch, prior.settlement_run_epoch + 1}
        ):
            raise SessionRunFenced("Model terminalization replay conflicts with durable state.")
        # Paired terminal evidence and exact release are authenticated by the
        # existing invocation cleanup boundary, including commit-before-ack.
        from cayu.runtime._invocation_lifecycle import require_released_invocation_command_authority
        from cayu.runtime.sessions import _activate_owned_session_run_fence

        if session.run_epoch == prior.settlement_run_epoch:
            owner = _activate_owned_session_run_fence(session)
            try:
                await engine._environment_lifecycle.release_run_fence_after_environment_cleanup(
                    session_id=session.id
                )
            finally:
                owner.retire()
            session = await store.load(request.session_id)
            if session is None:
                raise SessionRunFenced("Model terminalization session disappeared during release.")
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
        return ModelCompletionManualRecoveryResult(session=current, settlement=prior, replayed=True)
    active = await inspect_terminalization(store, session, checkpoint)
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
            model_recovery_id=source_id if outcome is InvocationTerminalOutcome.FAILED else None,
        )
        return updated

    coordinator = engine._recovery_coordinator
    claim = await coordinator._claim_incomplete_recovery(
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
        from cayu.runtime.sessions import (
            _incomplete_recovery_claim_from_checkpoint,
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
        terminal = engine._event_writer.prepare(
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
        return await engine._recover_model_completion_stage(
            request.model_copy(update={"expected_run_epoch": claim.session.run_epoch}),
            terminalization=(claim, decision, terminal),
        )

    failure = None
    try:
        async with asyncio.timeout(_TERMINALIZATION_DEADLINE_SECONDS):
            result = await coordinator._recover_incomplete_session_with_heartbeat(
                claim=claim,
                recovery=recover_owned,
            )
    except BaseException as exc:
        failure = exc
        raise
    finally:
        await coordinator._cleanup_incomplete_recovery_claim(
            authority=claim.require_authority(),
            authoritative_failure=failure,
        )
    current = await store.load(session.id)
    if current is None:
        raise SessionRunFenced("Model terminalization session disappeared after settlement.")
    return result.model_copy(update={"session": current})
