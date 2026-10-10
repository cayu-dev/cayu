"""Abandon an exact model stage that has not reached provider dispatch."""

from __future__ import annotations

import asyncio

from cayu._task_wait import (
    await_shielded_task_outcome,
    unexpected_child_cancellation_error,
)
from cayu._validation import (
    require_clean_nonblank,
)
from cayu.runtime._model_completion_contracts import (
    model_completion_recovery_context_from_stage,
)
from cayu.runtime._run_limits import (
    RunLimitController,
)
from cayu.sessions.base import (
    ModelCompletionStage,
    ModelCompletionStageAbandonmentResult,
    SessionStore,
)
from cayu.sessions.records import Session


async def abandon_pre_dispatch_model_stage(
    stage: ModelCompletionStage,
    *,
    session_store: SessionStore,
    run_limit_controller: RunLimitController,
    session: Session,
    authoritative_failure: BaseException,
    budget_dispatch_id: str | None = None,
) -> None:
    """Clear one provably undispatched stage without losing its root failure."""

    try:
        dispatch = await session_store.load_model_completion_stage_dispatch(
            session.id,
            stage.stage_id,
        )
        if dispatch is not None:
            authoritative_failure.add_note(
                "The prepared model-completion stage was retained because its exact "
                "dispatch receipt is durable."
            )
            return
        recovery_context = model_completion_recovery_context_from_stage(stage)
        await run_limit_controller.release_pre_provider_dispatch_reservations(
            reservation_ids=stage.reservation_ids,
            recovery_contexts=(
                () if recovery_context is None else recovery_context.budget_reservations
            ),
            dispatch_id=(
                stage.stage_id
                if budget_dispatch_id is None
                else require_clean_nonblank(budget_dispatch_id, "budget_dispatch_id")
            ),
        )
    except BaseException as release_error:
        authoritative_failure.add_note(
            "Pre-dispatch model-completion budget release also failed: "
            f"{type(release_error).__name__}: {release_error}"
        )
        return

    async def abandon_once() -> ModelCompletionStageAbandonmentResult:
        return await session_store.abandon_model_completion_stage(
            session.id,
            stage_id=stage.stage_id,
            preparation_digest=stage.preparation_digest,
            expected_run_epoch=session.run_epoch,
        )

    async def abandon_with_exact_replay() -> ModelCompletionStageAbandonmentResult:
        try:
            return await abandon_once()
        except (Exception, asyncio.CancelledError) as first_error:
            try:
                return await abandon_once()
            except (Exception, asyncio.CancelledError) as replay_error:
                replay_error.add_note(
                    "Exact model-completion stage abandonment replay also failed after "
                    f"{type(first_error).__name__}: {first_error}"
                )
                raise replay_error from first_error

    abandonment_task = asyncio.create_task(abandon_with_exact_replay())
    outcome = await await_shielded_task_outcome(
        abandonment_task,
        cancellation=(
            authoritative_failure
            if isinstance(authoritative_failure, asyncio.CancelledError)
            else None
        ),
    )
    abandonment_error = outcome.error
    if isinstance(abandonment_error, asyncio.CancelledError):
        abandonment_error = unexpected_child_cancellation_error(
            abandonment_error,
            operation="Pre-dispatch model-completion stage abandonment",
        )
    if abandonment_error is None:
        result = outcome.result
        try:
            if type(result) is not ModelCompletionStageAbandonmentResult:
                raise TypeError("Model-completion stage abandonment returned an invalid result.")
            abandonment = result.abandonment
            for field_name in (
                "session_id",
                "stage_id",
                "logical_step_id",
                "dispatch_ordinal",
                "purpose",
                "preparation_request_digest",
                "preparation_digest",
                "source_status",
                "source_run_epoch",
                "source_transcript_cursor",
            ):
                if getattr(abandonment, field_name) != getattr(stage, field_name):
                    raise RuntimeError(
                        "Model-completion stage abandonment acknowledged a different "
                        f"prepared stage field: {field_name}."
                    )
        except BaseException as validation_error:
            abandonment_error = validation_error
    if abandonment_error is not None:
        authoritative_failure.add_note(
            "Pre-dispatch model-completion stage abandonment also failed: "
            f"{type(abandonment_error).__name__}: {abandonment_error}"
        )
    if outcome.cancellation is not None and not isinstance(
        authoritative_failure, asyncio.CancelledError
    ):
        cancellation = outcome.cancellation
        cancellation.add_note(
            "Cancellation arrived while abandoning a model-completion stage after "
            f"{type(authoritative_failure).__name__}: {authoritative_failure}"
        )
        if abandonment_error is not None:
            cancellation.add_note(
                "Model-completion stage abandonment also failed: "
                f"{type(abandonment_error).__name__}: {abandonment_error}"
            )
        raise cancellation from authoritative_failure
