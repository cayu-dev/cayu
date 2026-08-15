"""Deep execution-profile admission rules shared by resume and recovery."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from cayu.runtime import _approval_support as approval_support
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _tool_round_recovery as tool_round_recovery
from cayu.runtime.execution_profiles import (
    ActiveInvocationExecutionProfile,
    ExecutionProfileComponentClass,
    ExecutionProfileIdentity,
    active_invocation_execution_profile_from_checkpoint,
    active_invocation_execution_profile_matches_session_epoch,
    build_execution_profile_identity,
    changed_execution_profile_components,
    execution_profile_with_component,
)
from cayu.runtime.sessions import Session
from cayu.runtime.user_input import pending_user_input_from_checkpoint
from cayu.vaults import SecretRedactor


@dataclass(frozen=True)
class ExecutionProfileContinuationPlan:
    """A reconstructed continuation and any component drift it exposes."""

    snapshot: ActiveInvocationExecutionProfile
    candidate_profile: ExecutionProfileIdentity
    changed_component_classes: tuple[ExecutionProfileComponentClass, ...]


def resolve_execution_profile_identity(
    *,
    registered_agent: runtime_records.RegisteredAgentState,
    provider_name: str,
    model: str,
    durable_system_prompt: str | None,
    runtime_name: str,
    runtime_version: str | None,
) -> ExecutionProfileIdentity:
    """Resolve one registered runtime body into its durable profile identity."""

    direct_tools = [
        {
            "name": tool.name,
            "description": tool.description,
            "schema": tool.schema,
            "parallel_safe": tool.parallel_safe,
            "effect": tool.effect.value,
            **({"workspace_mutation": True} if tool.workspace_mutation else {}),
        }
        for tool in registered_agent.tools.values()
    ]
    return build_execution_profile_identity(
        runtime_name=runtime_name,
        runtime_version=runtime_version,
        provider_name=provider_name,
        model=model,
        durable_system_prompt=durable_system_prompt,
        direct_tools=direct_tools,
    )


def prepare_execution_profile_continuation(
    *,
    session: Session,
    checkpoint: dict[str, Any] | None,
    registered_agent: runtime_records.RegisteredAgentState,
    registered_provider: runtime_records.RegisteredProvider,
    runtime_version: str | None,
    redactor: SecretRedactor,
    additional_profile_fingerprints: Iterable[str | None] = (),
) -> ExecutionProfileContinuationPlan:
    """Reconstruct a pending invocation and fail closed on invalid authority."""

    snapshot = active_invocation_execution_profile_from_checkpoint(checkpoint)
    if snapshot is None:
        raise RuntimeError(
            "Pending recovery state has no durable active invocation execution profile."
        )
    if not active_invocation_execution_profile_matches_session_epoch(
        snapshot,
        session_id=session.id,
        run_epoch=session.run_epoch,
    ):
        raise RuntimeError(
            "Active invocation execution profile does not match the session run epoch: "
            f"snapshot={snapshot.run_epoch}, session={session.run_epoch}."
        )
    pending_profile_fingerprints = {
        pending.execution_profile_fingerprint
        for pending in (
            tool_round_recovery.pending_tool_round_from_checkpoint(
                checkpoint,
                redactor=redactor,
                consume_on_rejection=True,
            ),
            approval_support.pending_approval_from_checkpoint(
                checkpoint,
                redactor=redactor,
                consume_on_rejection=True,
            ),
            pending_user_input_from_checkpoint(
                checkpoint,
                redactor=redactor,
                consume_on_rejection=True,
            ),
        )
        if pending is not None
    }
    pending_profile_fingerprints.update(additional_profile_fingerprints)
    if pending_profile_fingerprints and pending_profile_fingerprints != {
        snapshot.profile.fingerprint
    }:
        raise RuntimeError(
            "Pending recovery state does not reference the active invocation execution profile."
        )
    candidate = resolve_execution_profile_identity(
        registered_agent=registered_agent,
        provider_name=registered_provider.name,
        model=session.model,
        durable_system_prompt=None,
        runtime_name="cayu",
        runtime_version=runtime_version,
    )
    candidate = execution_profile_with_component(
        candidate,
        snapshot.profile.component(ExecutionProfileComponentClass.DURABLE_SYSTEM_PROJECTION),
    )
    return ExecutionProfileContinuationPlan(
        snapshot=snapshot,
        candidate_profile=candidate,
        changed_component_classes=changed_execution_profile_components(
            snapshot.profile,
            candidate,
        ),
    )
