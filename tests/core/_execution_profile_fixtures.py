from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any
from uuid import uuid4

from cayu.core.events import Event
from cayu.runtime import _session_request_boundary as session_request_boundary
from cayu.runtime import _transcript as transcript_helpers
from cayu.runtime.app import CayuApp
from cayu.runtime.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
    decode_runtime_checkpoint,
)
from cayu.runtime.execution_profiles import (
    ActiveInvocationExecutionProfile,
    active_invocation_execution_profile_from_checkpoint,
    build_execution_profile_identity,
    checkpoint_with_active_invocation_execution_profile,
    execution_profile_from_session_metadata,
)
from cayu.runtime.sessions import (
    RunRequest,
    Session,
    SessionIdentity,
    SessionStore,
    bind_runtime_session_create_claim,
)
from cayu.vaults import SecretRedactor


@dataclass(frozen=True)
class AdmittedSessionFixture:
    """A new session carrying the complete runtime-owned admission state."""

    request: RunRequest
    session: Session
    identity: SessionIdentity
    interaction_started_event: Event
    active_invocation_profile: ActiveInvocationExecutionProfile


async def create_admitted_session(
    store: SessionStore,
    *,
    request: RunRequest,
    provider_name: str,
    model: str,
    durable_system_prompt: str | None = None,
    direct_tools: Iterable[Mapping[str, Any]] = (),
    interaction_id: str | None = None,
    secret_redactor: SecretRedactor | None = None,
) -> AdmittedSessionFixture:
    """Create the production-valid starting point for resume/recovery tests."""

    app = CayuApp(
        session_store=store,
        enable_logging=False,
        secret_redactor=secret_redactor,
    )
    prepared_request = session_request_boundary.prepare_run_request(
        request,
        redactor=app._secret_redactor,
    )
    if prepared_request.session_id is None:
        prepared_request = prepared_request.model_copy(update={"session_id": str(uuid4())})
    session_id = prepared_request.session_id
    if session_id is None:
        raise AssertionError("Admitted-session fixture failed to assign a session identity.")
    if interaction_id is None:
        interaction_id = str(uuid4())

    identity = profiled_session_identity(
        provider_name=provider_name,
        model=model,
        durable_system_prompt=durable_system_prompt,
        direct_tools=direct_tools,
    )
    execution_profile = identity.execution_profile
    if execution_profile is None:
        raise AssertionError("Admitted-session fixture failed to build an execution profile.")
    started_event = runtime_interaction_started_event(
        app,
        session_id=session_id,
        interaction_id=interaction_id,
        agent_name=prepared_request.agent_name,
        environment_name=prepared_request.environment_name,
    )
    bind_runtime_session_create_claim(
        prepared_request,
        identity=identity,
        interaction_started_event=started_event,
    )

    def freeze_initial_invocation_profile(
        current_session: Session,
        checkpoint: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return checkpoint_with_active_invocation_execution_profile(
            checkpoint,
            session_id=current_session.id,
            interaction_id=interaction_id,
            run_epoch=current_session.run_epoch,
            profile=execution_profile,
        )

    runtime_store = app._runtime_session_store
    await runtime_store.create(
        prepared_request,
        identity=identity,
        interaction_started_event=started_event,
        interaction_source_messages=prepared_request.messages,
        checkpoint_transform=freeze_initial_invocation_profile,
    )
    await runtime_store.replace_initial_transcript_messages(
        session_id,
        prepared_request.messages,
        transcript_helpers.initial_messages(
            system_prompt=durable_system_prompt,
            request_messages=prepared_request.messages,
        ),
        interaction_id=interaction_id,
    )
    checkpoint = await runtime_store.load_checkpoint(session_id)
    active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
    if active_profile is None:
        raise AssertionError("Admitted-session fixture lost active invocation authority.")
    refreshed_session = await runtime_store.load(session_id)
    if refreshed_session is None:
        raise AssertionError("Admitted-session fixture lost the created session.")
    return AdmittedSessionFixture(
        request=prepared_request,
        session=refreshed_session,
        identity=identity,
        interaction_started_event=started_event,
        active_invocation_profile=active_profile,
    )


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
