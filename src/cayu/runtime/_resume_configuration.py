"""Inherit ordinary-resume controls from verified durable invocation evidence."""

from __future__ import annotations

from typing import Any

from cayu.context.structured_output import StructuredOutputSpec, copy_structured_output_spec
from cayu.execution_profiles import (
    ExecutionProfileComponentClass,
    ExecutionProfileComponentIdentity,
    ExecutionProfileIdentity,
    ExecutionProfileIdentityStrength,
    _available_component,
)
from cayu.providers.retry_policy import RetryPolicy
from cayu.runtime._execution_profile_admission import model_finalization_material
from cayu.runtime._execution_profile_continuation import _execution_profile_structured_output
from cayu.runtime._model_completion_contracts import (
    ModelCompletionRecoveryContext,
    model_completion_recovery_context_from_stage,
)
from cayu.runtime.work_attempt_semantics import WorkAttemptRunSemantics
from cayu.sessions._execution_profile_checkpoint import (
    active_invocation_execution_profile_from_checkpoint,
    active_invocation_execution_profile_matches_session_epoch,
    execution_profile_from_session_metadata,
)
from cayu.sessions._invocation_configuration import (
    INVOCATION_CONFIGURATION_CHECKPOINT_KEY,
    InvocationConfiguration,
)
from cayu.sessions._model_completion_publication import model_step_publication_from_checkpoint
from cayu.sessions.records import Session
from cayu.sessions.requests import ResumeRequest, RunRequest
from cayu.sessions.store import SessionStore


def _structured_component(spec: StructuredOutputSpec | None) -> ExecutionProfileComponentIdentity:
    material = _execution_profile_structured_output(spec)
    return _available_component(
        ExecutionProfileComponentClass.STRUCTURED_OUTPUT,
        ExecutionProfileIdentityStrength.STRUCTURAL,
        {"kind": "none", "version": 1} if material is None else material,
    )


def invocation_configuration(
    request: RunRequest | ResumeRequest | WorkAttemptRunSemantics,
    *,
    session_id: str,
    session_instance_id: str,
    profile: ExecutionProfileIdentity,
    retry_policy: RetryPolicy,
) -> InvocationConfiguration:
    return InvocationConfiguration(
        session_id=session_id,
        session_instance_id=session_instance_id,
        execution_profile_fingerprint=profile.fingerprint,
        structured_output=request.structured_output,
        max_steps=request.max_steps,
        limits=request.limits,
        retry_policy=retry_policy,
        tool_completion=request.tool_completion,
    )


def _finalization_component(
    source: InvocationConfiguration | ModelCompletionRecoveryContext,
) -> ExecutionProfileComponentIdentity:
    return _available_component(
        ExecutionProfileComponentClass.FINALIZATION,
        ExecutionProfileIdentityStrength.STRUCTURAL,
        model_finalization_material(
            max_steps=source.max_steps,
            limits=source.limits,
            retry_policy=source.retry_policy,
            tool_completion=source.tool_completion,
        ),
    )


def _configuration_matches_profile(
    record: InvocationConfiguration, profile: ExecutionProfileIdentity
) -> bool:
    return (
        record.execution_profile_fingerprint == profile.fingerprint
        and profile.component(ExecutionProfileComponentClass.STRUCTURED_OUTPUT)
        == _structured_component(record.structured_output)
        and profile.component(ExecutionProfileComponentClass.FINALIZATION)
        == _finalization_component(record)
    )


def checkpoint_with_invocation_configuration(
    checkpoint: dict[str, Any] | None,
    *,
    session: Session,
    profile: ExecutionProfileIdentity,
    configuration: InvocationConfiguration | None,
) -> dict[str, Any]:
    """Publish settings inside the same transaction that admits their profile."""
    updated = dict(checkpoint or {})
    updated.pop(INVOCATION_CONFIGURATION_CHECKPOINT_KEY, None)
    if configuration is not None:
        if (
            configuration.session_id != session.id
            or configuration.session_instance_id != session.instance_id
            or not _configuration_matches_profile(configuration, profile)
        ):
            raise ValueError("Invocation configuration does not match its admitted profile.")
        updated[INVOCATION_CONFIGURATION_CHECKPOINT_KEY] = configuration.model_dump(mode="json")
    return updated


def recorded_admitted_configuration(
    checkpoint: dict[str, Any] | None,
    session: Session,
) -> InvocationConfiguration | None:
    if not checkpoint:
        return None
    raw = checkpoint.get(INVOCATION_CONFIGURATION_CHECKPOINT_KEY)
    if raw is None:
        return None
    record = InvocationConfiguration.model_validate(raw)
    active = active_invocation_execution_profile_from_checkpoint(checkpoint)
    expected = execution_profile_from_session_metadata(session.metadata)
    if (
        record.session_id != session.id
        or record.session_instance_id != session.instance_id
        or active is None
        or not active_invocation_execution_profile_matches_session_epoch(
            active, session_id=session.id, run_epoch=session.run_epoch
        )
        or not _configuration_matches_profile(record, expected)
        or not _configuration_matches_profile(record, active.profile)
    ):
        return None
    return record


def _inherit_admitted_configuration(
    checkpoint: dict[str, Any] | None,
    session: Session,
    request: ResumeRequest,
) -> ResumeRequest:
    record = recorded_admitted_configuration(checkpoint, session)
    if record is None:
        return request
    fields = request.model_fields_set
    inherited = request.model_copy(
        update={
            name: getattr(record, name)
            for name in ("structured_output", "max_steps", "limits", "retry_policy")
            if name not in fields
        },
        # The record owns detached data; request collaborators and authenticated
        # private handoffs must retain the identities preserved by its copier.
        deep=False,
    )
    object.__setattr__(inherited, "__pydantic_fields_set__", set(fields))
    return inherited


async def recorded_invocation_semantics(
    store: SessionStore,
    session: Session,
    checkpoint: dict[str, Any] | None,
) -> ModelCompletionRecoveryContext | None:
    """Return the latest published model step's run semantics for the active invocation.

    The context is returned only when it belongs to the session's current active
    invocation and its exact profile; otherwise ``None``. It is read-only.
    """
    pointer = model_step_publication_from_checkpoint(checkpoint)
    active = active_invocation_execution_profile_from_checkpoint(checkpoint)
    if pointer is None or active is None:
        return None
    stage = await store.load_model_completion_stage(session.id, pointer.stage_id)
    if stage is None or stage.logical_step_id != pointer.logical_step_id:
        return None
    context = model_completion_recovery_context_from_stage(stage)
    if (
        context is None
        or not active_invocation_execution_profile_matches_session_epoch(
            active, session_id=session.id, run_epoch=session.run_epoch
        )
        or context.interaction_id != active.interaction_id
        or context.execution_profile_fingerprint != active.profile.fingerprint
    ):
        return None
    return context


async def inherit_resume_configuration(
    store: SessionStore, session: Session, request: ResumeRequest, default_retry: RetryPolicy
) -> tuple[ResumeRequest, tuple[str, ...]]:
    """Return detached controls and safe diagnostics; never infer values from hashes.

    Requests without matching durable settings retain their declared defaults.
    Admission still checks the full profile and current resource authority afterward.
    """
    checkpoint = await store.load_checkpoint(session.id)
    request = _inherit_admitted_configuration(checkpoint, session, request)
    context = await recorded_invocation_semantics(store, session, checkpoint)
    if context is None:
        return request, ()
    active = active_invocation_execution_profile_from_checkpoint(checkpoint)
    assert active is not None
    expected = execution_profile_from_session_metadata(session.metadata)
    component = _finalization_component(context)
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
    structured_component = _structured_component(context.structured_output)
    if "structured_output" not in fields and all(
        profile.component(ExecutionProfileComponentClass.STRUCTURED_OUTPUT) == structured_component
        for profile in (expected, active.profile)
    ):
        updates["structured_output"] = copy_structured_output_spec(context.structured_output)
    pending_continuation = any(
        checkpoint is not None and checkpoint.get(key) is not None
        for key in ("pending_tool_round", "pending_tool_approval", "pending_user_input")
    )
    if pending_continuation:
        if "tool_completion" not in fields:
            updates["tool_completion"] = context.tool_completion
        elif request.tool_completion != context.tool_completion:
            raise ValueError("tool_completion cannot change during a pending continuation.")
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
