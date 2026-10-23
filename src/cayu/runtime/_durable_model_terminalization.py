"""Terminal-only model recovery under the existing incomplete-recovery claim.

Executable registrations never enter this path. Unknown dependent work blocks it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from cayu.runtime._model_completion_contracts import (
    ModelCompletionManualRecoveryRequired,
    model_completion_recovery_context_from_stage,
)
from cayu.sessions._durable_operation_ownership import DurableOperationOwnership
from cayu.sessions._execution_profile_checkpoint import (
    ActiveInvocationExecutionProfile,
    active_invocation_execution_profile_from_checkpoint,
)
from cayu.sessions._model_stage import (
    ActiveModelCompletionStage,
    ModelCompletionStage,
)
from cayu.sessions._model_stage_rules import (
    _validate_model_completion_stage_dispatch,
)
from cayu.sessions.authority import SessionRunFenced
from cayu.sessions.invocation import SessionExecutionSource
from cayu.sessions.queries import SessionQuery
from cayu.sessions.records import Session
from cayu.sessions.store import SessionStore

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


_ALLOWED_KEYS = {
    "checkpoint_schema_version",
    "active_invocation_execution_profile",
    "invocation_lifecycle_receipt",
    "invocation_configuration",
    "invocation_terminal_decision",
    "settled_invocation_terminal_decision",
    "incomplete_session_recovery_claim",
    "recovery_plan_execution",
}


def _completed_step_pointer_key(checkpoint: dict[str, Any] | None) -> set[str]:
    """Admit the pointer to an earlier, fully published step of the same turn.

    A turn that crashes in a later model call still records its last published
    step. The deferred flag describes its original publication, not whether its
    tool round is still open. Pending work is excluded by the checkpoint keys;
    inspect_terminalization also requires durable proof that a tool round closed.
    """

    from cayu.sessions._model_completion_publication import (
        LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY,
        model_step_publication_from_checkpoint,
    )

    pointer = model_step_publication_from_checkpoint(checkpoint)
    if pointer is None:
        return set()
    return {LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY}


def require_terminalization_checkpoint(
    session: Session, checkpoint: dict[str, Any] | None
) -> ActiveInvocationExecutionProfile:
    if (
        session.environment_name is not None
        or session.parent_session_id is not None
        or session.invocation.source
        not in {SessionExecutionSource.SDK_RUN, SessionExecutionSource.HTTP_RUN}
        or not checkpoint
        or set(checkpoint) - _ALLOWED_KEYS - _completed_step_pointer_key(checkpoint)
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
    await _require_completed_tool_publication(store, session, checkpoint, stage)
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


async def _require_completed_tool_publication(
    store: SessionStore,
    session: Session,
    checkpoint: dict[str, Any] | None,
    stage: ModelCompletionStage,
) -> None:
    """Require immutable closure evidence for an earlier deferred tool step."""
    from cayu.sessions._model_completion_publication import model_step_publication_from_checkpoint

    pointer = model_step_publication_from_checkpoint(checkpoint)
    if pointer is None or pointer.tool_round_id is None:
        return
    publication_id = f"tool-round:{pointer.tool_round_id}"
    receipt = await store.load_runtime_publication_receipt(session.id, publication_id)
    expected_end = pointer.transcript_end_cursor + (2 if pointer.assistant_message_deferred else 1)
    if (
        receipt is None
        or receipt.publication_id != publication_id
        or receipt.kind != "tool-round"
        or receipt.transcript_start_cursor != pointer.transcript_end_cursor
        or receipt.transcript_end_cursor != expected_end
        or receipt.intent.get("round_id") != pointer.tool_round_id
        or receipt.intent.get("tool_round_id") != pointer.tool_round_id
        or receipt.intent.get("model_step_id") != pointer.logical_step_id
        or stage.logical_step_id == pointer.logical_step_id
        or stage.source_transcript_cursor < expected_end
    ):
        raise ModelCompletionManualRecoveryRequired(
            "Model terminalization requires durable closure of the preceding tool round."
        )
