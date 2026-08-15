from __future__ import annotations

from collections.abc import Iterable, Mapping
from importlib.metadata import version
from typing import Any

from cayu.core.events import Event
from cayu.runtime.app import CayuApp
from cayu.runtime.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
    decode_runtime_checkpoint,
)
from cayu.runtime.execution_profiles import (
    active_invocation_execution_profile_from_checkpoint,
    build_execution_profile_identity,
    checkpoint_with_active_invocation_execution_profile,
    execution_profile_from_session_metadata,
)
from cayu.runtime.sessions import Session, SessionIdentity


def runtime_interaction_started_event(
    app: CayuApp,
    *,
    session_id: str,
    interaction_id: str,
    agent_name: str,
    environment_name: str | None = None,
) -> Event:
    """Build exact runtime interaction evidence for low-level crash fixtures."""

    return app._session_engine._interaction_started_event_from_identity(
        session_id=session_id,
        interaction_id=interaction_id,
        agent_name=agent_name,
        environment_name=environment_name,
    )


def profiled_session_identity(
    *,
    provider_name: str,
    model: str,
    durable_system_prompt: str | None = None,
    direct_tools: Iterable[Mapping[str, Any]] = (),
) -> SessionIdentity:
    """Build the identity used by low-level tests that later enter public resume."""

    runtime_version = version("cayu")
    return SessionIdentity(
        provider_name=provider_name,
        model=model,
        runtime_name="cayu",
        runtime_version=runtime_version,
        execution_profile=build_execution_profile_identity(
            runtime_name="cayu",
            runtime_version=runtime_version,
            provider_name=provider_name,
            model=model,
            durable_system_prompt=durable_system_prompt,
            direct_tools=direct_tools,
        ),
    )


def checkpoint_with_rebound_test_invocation_profile(
    session: Session,
    checkpoint: dict[str, Any] | None,
    *,
    interaction_id: str | None = None,
) -> dict[str, Any]:
    """Atomically model a test worker claim under the session's durable profile."""

    decoded = decode_runtime_checkpoint(checkpoint, session_id=session.id)
    if decoded is None:
        decoded = {CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION}
    active_profile = active_invocation_execution_profile_from_checkpoint(decoded)
    if active_profile is None:
        profile = execution_profile_from_session_metadata(session.metadata)
        if interaction_id is None:
            raise AssertionError("A first test invocation claim requires an interaction id.")
    else:
        profile = active_profile.profile
        if interaction_id is None:
            interaction_id = active_profile.interaction_id
    return checkpoint_with_active_invocation_execution_profile(
        decoded,
        session_id=session.id,
        interaction_id=interaction_id,
        run_epoch=session.run_epoch + 1,
        profile=profile,
        expected=active_profile,
    )
