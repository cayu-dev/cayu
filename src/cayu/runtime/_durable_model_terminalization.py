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
