"""Compaction-specific accounting and checkpoint decisions during stage recovery."""

from __future__ import annotations

from cayu.events import Event
from cayu.execution_units import ModelAttemptIdentity
from cayu.runtime._model_completion_contracts import (
    ModelCompletionManualRecoveryRequired,
    model_completion_recovery_context_from_stage,
)
from cayu.runtime._run_limits import BorrowedAutomaticCompactionOutcomeUnknown, RunLimitController
from cayu.sessions.base import ModelCompletionStage
from cayu.sessions.records import Session


async def reconcile_before_stage_recovery(
    *,
    session: Session,
    stage: ModelCompletionStage,
    run_limit_controller: RunLimitController,
) -> tuple[Event, ...]:
    """Resolve compaction spend attached to the stage before recovery advances it."""

    try:
        return tuple(
            await run_limit_controller.reconcile_borrowed_automatic_compaction_budget_authority(
                session=session,
                stage=stage,
            )
        )
    except BorrowedAutomaticCompactionOutcomeUnknown as outcome_unknown:
        raise ModelCompletionManualRecoveryRequired(str(outcome_unknown)) from outcome_unknown


async def reconcile_completed_stage(
    *,
    session: Session,
    stage: ModelCompletionStage,
    run_limit_controller: RunLimitController,
) -> list[Event]:
    """Settle a completed compactor dispatch under its saved budget authority."""

    if stage.purpose != "context-compaction":
        raise ValueError("Compaction recovery requires a context-compaction stage.")
    if not stage.reservation_ids:
        return []
    recovery_context = model_completion_recovery_context_from_stage(stage)
    pricing_provider_name = stage.intent.get("pricing_provider_name")
    requested_model = stage.intent.get("requested_model")
    model_attempt_id = stage.intent.get("model_attempt_id")
    if recovery_context is None or not all(
        type(value) is str
        for value in (
            pricing_provider_name,
            requested_model,
            model_attempt_id,
        )
    ):
        raise ModelCompletionManualRecoveryRequired(
            "Completed context-compaction recovery lost its exact budget authority."
        )
    assert isinstance(pricing_provider_name, str)
    assert isinstance(requested_model, str)
    assert isinstance(model_attempt_id, str)
    return await run_limit_controller.reconcile_completed_automatic_compaction_reservations(
        session=session,
        stage=stage,
        recovery_contexts=recovery_context.budget_reservations,
        pricing_provider_name=pricing_provider_name,
        model=requested_model,
        model_attempt_identity=ModelAttemptIdentity(
            model_step_id=stage.logical_step_id,
            model_attempt_id=model_attempt_id,
        ),
    )


def require_promoted_context_checkpoint(stage: ModelCompletionStage) -> None:
    """Reject a promoted compaction whose context checkpoint is unavailable."""

    if stage.purpose == "context-compaction":
        raise ModelCompletionManualRecoveryRequired(
            "The completed context compaction was promoted without a durable "
            "context checkpoint; its completion evidence prevents provider "
            "redispatch."
        )
