"""Inherit ordinary-resume controls from exact durable model-step evidence."""

from __future__ import annotations

from cayu.runtime._execution_profile_admission import model_finalization_material
from cayu.runtime._model_completion_publication import model_step_publication_from_checkpoint
from cayu.runtime._model_step_executor import model_completion_recovery_context_from_stage
from cayu.runtime.execution_profiles import (
    ExecutionProfileComponentClass,
    ExecutionProfileIdentityStrength,
    _available_component,
    active_invocation_execution_profile_from_checkpoint,
    active_invocation_execution_profile_matches_session_epoch,
    execution_profile_from_session_metadata,
)
from cayu.runtime.retry_policy import RetryPolicy
from cayu.sessions.base import ResumeRequest, Session, SessionStore


async def inherit_resume_configuration(
    store: SessionStore, session: Session, request: ResumeRequest, default_retry: RetryPolicy
) -> tuple[ResumeRequest, tuple[str, ...]]:
    """Return detached controls and safe diagnostics; never infer values from hashes.

    Legacy/fork sessions without a matching completed step retain existing defaults.
    Admission still checks the full profile and current resource authority afterward.
    """
    checkpoint = await store.load_checkpoint(session.id)
    pointer = model_step_publication_from_checkpoint(checkpoint)
    active = active_invocation_execution_profile_from_checkpoint(checkpoint)
    if pointer is None or active is None:
        return request, ()
    stage = await store.load_model_completion_stage(session.id, pointer.stage_id)
    if stage is None or stage.logical_step_id != pointer.logical_step_id:
        return request, ()
    context = model_completion_recovery_context_from_stage(stage)
    if (
        context is None
        or not active_invocation_execution_profile_matches_session_epoch(
            active, session_id=session.id, run_epoch=session.run_epoch
        )
        or context.interaction_id != active.interaction_id
        or context.execution_profile_fingerprint != active.profile.fingerprint
    ):
        return request, ()
    expected = execution_profile_from_session_metadata(session.metadata)
    component = _available_component(
        ExecutionProfileComponentClass.FINALIZATION,
        ExecutionProfileIdentityStrength.STRUCTURAL,
        model_finalization_material(
            max_steps=context.max_steps, limits=context.limits, retry_policy=context.retry_policy
        ),
    )
    if any(
        profile.component(ExecutionProfileComponentClass.FINALIZATION) != component
        for profile in (expected, active.profile)
    ):
        return request, ()
    fields = request.model_fields_set
    updates = {
        name: getattr(context, name)
        for name in ("max_steps", "limits", "retry_policy")
        if name not in fields
    }
    resolved = request.model_copy(update=updates)
    # Do not turn inherited values into caller-supplied overrides.
    object.__setattr__(resolved, "__pydantic_fields_set__", set(fields))
    differences = []
    if resolved.max_steps != context.max_steps:
        differences.append(f"max_steps: stored={context.max_steps}, requested={resolved.max_steps}")
    for field, stored_value in context.limits.model_dump().items():
        requested_value = getattr(resolved.limits, field)
        if requested_value != stored_value:
            differences.append(
                f"limits.{field}: stored={stored_value}, requested={requested_value}"
            )
    retry = resolved.retry_policy or default_retry
    for field, stored_value in context.retry_policy.model_dump().items():
        requested_value = getattr(retry, field)
        if requested_value != getattr(context.retry_policy, field):
            differences.append(
                f"retry_policy.{field}: stored={stored_value}, requested={requested_value}"
            )
    return resolved, tuple(differences)
