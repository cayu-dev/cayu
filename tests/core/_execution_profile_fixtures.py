from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any
from uuid import uuid4

from cayu.core.agents import AgentSpec
from cayu.core.events import Event, EventType, event_with_runtime_envelope_authority
from cayu.core.execution_identity import (
    ExecutionProfileBehaviorIdentity,
    copy_execution_profile_behavior_identity,
)
from cayu.core.messages import Message
from cayu.core.thinking import ThinkingConfig, thinking_config_payload
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.evals.testing import ScriptedModelProvider
from cayu.providers import ModelProvider
from cayu.runtime import _execution_profile_admission as execution_profile_admission
from cayu.runtime import _session_engine as session_engine_module
from cayu.runtime import _session_request_boundary as session_request_boundary
from cayu.runtime import _transcript as transcript_helpers
from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store
from cayu.runtime._invocation_lifecycle import (
    AdmitInvocationCommand,
    CreateInvocationCommand,
    InvocationCheckpointPatch,
    InvocationMutationResult,
    InvocationReleaseResult,
    ReleaseInvocationCommand,
    _release_invocation_command_with_cleanup_authority,
    invocation_checkpoint_state_sha256,
    prepare_rebind_invocation_command,
)
from cayu.runtime.app import CayuApp
from cayu.runtime.budgets import BudgetLimit, request_budget_limits_for_session
from cayu.runtime.build_provenance import current_runtime_build_provenance
from cayu.runtime.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
    INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
    decode_runtime_checkpoint,
)
from cayu.runtime.execution_profiles import (
    ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
    ActiveInvocationExecutionProfile,
    ExecutionProfileDecision,
    ExecutionProfileIdentity,
    active_invocation_execution_profile_from_checkpoint,
    checkpoint_with_active_invocation_execution_profile,
    execution_profile_from_session_metadata,
)
from cayu.runtime.hooks import RuntimeHook
from cayu.runtime.loop_policies import LoopPolicy
from cayu.runtime.retry_policy import RetryPolicy, copy_retry_policy
from cayu.runtime.sessions import (
    RunRequest,
    Session,
    SessionIdentity,
    SessionModelTransition,
    SessionStatus,
    SessionStore,
    bind_runtime_session_create_claim,
    run_request_with_runtime_session_instance_authority,
    runtime_publication_checkpoint_mutation,
)
from cayu.runtime.stop_policy import RunLimits, copy_run_limits
from cayu.runtime.structured_output import StructuredOutputSpec
from cayu.runtime.tool_exposure import (
    ToolCapabilityCeiling,
    ToolExposurePolicy,
    tool_capability_ceiling_from_session_metadata,
)
from cayu.runtime.tool_policy import ToolPolicy
from cayu.vaults import SecretRedactor


class _ProfileFixtureTool(Tool):
    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        del ctx, args
        raise AssertionError("Execution-profile fixture tools must never run.")


def versioned_test_provider_identity(
    provider: ModelProvider,
    *,
    behavior_version: str = "1",
) -> ExecutionProfileBehaviorIdentity:
    """Return a stable test-owned identity for one declared provider behavior.

    Restart and rolling-deployment tests use this helper to make their custom
    adapters explicitly reconstructable. Tests for opaque-provider rejection
    deliberately continue to inherit ``ModelProvider`` without using it.
    """

    provider_type = type(provider)
    return ExecutionProfileBehaviorIdentity(
        name=f"tests:{provider_type.__module__}.{provider_type.__qualname__}",
        behavior_version=behavior_version,
        implementation_version="1",
    )


def tool_capability_ceiling_for_app(
    app: CayuApp,
    *,
    agent_name: str = "assistant",
) -> ToolCapabilityCeiling:
    """Return the complete direct-tool ceiling for a registered test agent."""

    return ToolCapabilityCeiling(tool_names=tuple(app._agents[agent_name].tools))


def run_request_with_registered_tool_ceiling(
    app: CayuApp,
    **request_fields: Any,
) -> RunRequest:
    """Build a request carrying the registered agent's complete tool ceiling."""

    request = RunRequest(**request_fields)
    return request.model_copy(
        update={
            "tool_capability_ceiling": tool_capability_ceiling_for_app(
                app,
                agent_name=request.agent_name,
            )
        }
    )


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
    tools: Iterable[Tool] | None = None,
    execution_profile: ExecutionProfileIdentity | None = None,
    interaction_id: str | None = None,
    interaction_started_event: Event | None = None,
    secret_redactor: SecretRedactor | None = None,
    provider: ModelProvider | None = None,
    app: CayuApp | None = None,
) -> AdmittedSessionFixture:
    """Create the production-valid starting point for resume/recovery tests."""

    direct_tool_material = tuple(direct_tools)
    resolved_tools = None if tools is None else tuple(tools)
    if app is None:
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            secret_redactor=secret_redactor,
        )
    elif secret_redactor is not None:
        raise ValueError("secret_redactor and app are mutually exclusive.")
    prepared_request = session_request_boundary.prepare_run_request(
        request,
        redactor=app._secret_redactor,
    )
    if prepared_request.tool_capability_ceiling is None:
        if resolved_tools is not None:
            ceiling_names = tuple(tool.name for tool in resolved_tools)
        else:
            ceiling_names = tuple(str(item["name"]) for item in direct_tool_material)
        prepared_request = prepared_request.model_copy(
            update={
                "tool_capability_ceiling": ToolCapabilityCeiling(
                    tool_names=ceiling_names,
                )
            }
        )
    if prepared_request.session_id is None:
        prepared_request = prepared_request.model_copy(update={"session_id": str(uuid4())})
    session_id = prepared_request.session_id
    if session_id is None:
        raise AssertionError("Admitted-session fixture failed to assign a session identity.")
    if interaction_id is None:
        interaction_id = (
            str(uuid4())
            if interaction_started_event is None
            else interaction_started_event.interaction_id
        )
    if interaction_id is None:
        raise ValueError("interaction_started_event must carry an interaction identity.")

    identity = profiled_session_identity(
        provider_name=provider_name,
        model=model,
        durable_system_prompt=durable_system_prompt,
        direct_tools=direct_tool_material,
        tools=resolved_tools,
        invocation_loop_policies=prepared_request.loop_policies,
        execution_profile=execution_profile,
        provider=provider,
        app=(app if execution_profile is None and app._providers else None),
        agent_name=prepared_request.agent_name,
        structured_output=prepared_request.structured_output,
        thinking=prepared_request.thinking,
        request_budget_limits=prepared_request.budget_limits,
        causal_budget_id=(
            prepared_request.causal_budget_id or prepared_request.task_id or session_id
        ),
        max_steps=prepared_request.max_steps,
        limits=prepared_request.limits,
        retry_policy=app._effective_retry_policy(prepared_request.retry_policy),
    )
    execution_profile = identity.execution_profile
    if execution_profile is None:
        raise AssertionError("Admitted-session fixture failed to build an execution profile.")
    started_event = (
        runtime_interaction_started_event(
            app,
            session_id=session_id,
            interaction_id=interaction_id,
            agent_name=prepared_request.agent_name,
            environment_name=prepared_request.environment_name,
        )
        if interaction_started_event is None
        else interaction_started_event
    )
    session_instance_id = str(uuid4())
    prepared_request = run_request_with_runtime_session_instance_authority(
        prepared_request,
        session_instance_id=session_instance_id,
    )
    bind_runtime_session_create_claim(
        prepared_request,
        identity=identity,
        interaction_started_event=started_event,
    )

    runtime_store = app._runtime_session_store
    active_profile = ActiveInvocationExecutionProfile(
        session_id=session_id,
        interaction_id=interaction_id,
        run_epoch=1,
        profile=execution_profile,
    )
    await runtime_store.apply_invocation_lifecycle_command(
        CreateInvocationCommand(
            session_id=session_id,
            expected_session_instance_id=session_instance_id,
            request=prepared_request,
            identity=identity,
            active_profile=active_profile,
            interaction_started_event=started_event,
            interaction_source_messages=tuple(prepared_request.messages),
        )
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
    stored_active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
    if stored_active_profile is None:
        raise AssertionError("Admitted-session fixture lost active invocation authority.")
    refreshed_session = await runtime_store.load(session_id)
    if refreshed_session is None:
        raise AssertionError("Admitted-session fixture lost the created session.")
    return AdmittedSessionFixture(
        request=prepared_request,
        session=refreshed_session,
        identity=identity,
        interaction_started_event=started_event,
        active_invocation_profile=stored_active_profile,
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
    agent_name: str = "assistant",
    durable_system_prompt: str | None = None,
    direct_tools: Iterable[Mapping[str, Any]] = (),
    tools: Iterable[Tool] | None = None,
    tool_policy: ToolPolicy | None = None,
    tool_exposure_policy: ToolExposurePolicy | None = None,
    runtime_hooks: Iterable[RuntimeHook] = (),
    invocation_loop_policies: Iterable[LoopPolicy] = (),
    execution_profile: ExecutionProfileIdentity | None = None,
    provider: ModelProvider | None = None,
    structured_output: StructuredOutputSpec | None = None,
    thinking: ThinkingConfig | None = None,
    request_budget_limits: tuple[BudgetLimit, ...] = (),
    causal_budget_id: str | None = None,
    max_steps: int = 16,
    limits: RunLimits | None = None,
    retry_policy: RetryPolicy | None = None,
    app: CayuApp | None = None,
) -> SessionIdentity:
    """Build the identity used by low-level tests that later enter public resume."""

    runtime_version = version("cayu")
    resolved_profile = execution_profile
    if resolved_profile is None:
        if app is not None:
            registered_agent = app._agents[agent_name]
            resolved_profile = session_engine_module._execution_profile_identity(
                registered_agent=registered_agent,
                provider_name=provider_name,
                registered_provider=app._providers.get(provider_name),
                model=model,
                durable_system_prompt=durable_system_prompt,
                redactor=app._secret_redactor,
                process_identity=app._execution_profile_process_identity,
                runtime_hooks=app._runtime_hooks,
                loop_policies=app._loop_policies,
                loop_policy_execution_profile_identities=(
                    app._loop_policy_execution_profile_identities
                ),
                budget_policy=app.budget_policy,
                request_budget_limits=request_budget_limits,
                causal_budget_id=causal_budget_id,
                structured_output=structured_output,
                thinking=thinking,
                max_steps=max_steps,
                limits=limits,
                retry_policy=retry_policy,
            )
            return SessionIdentity(
                provider_name=provider_name,
                model=model,
                runtime_name="cayu",
                runtime_version=runtime_version,
                execution_profile=resolved_profile,
            )
        tool_material = tuple(direct_tools)
        resolved_tools = (
            tuple(tools)
            if tools is not None
            else tuple(
                _ProfileFixtureTool(
                    ToolSpec(
                        name=str(item["name"]),
                        description=str(item.get("description", "")),
                        input_schema=dict(item.get("schema", {})),
                        parallel_safe=item.get("parallel_safe", True),
                        effect=item.get("effect", "external"),
                        workspace_mutation=item.get("workspace_mutation", False),
                    )
                )
                for item in tool_material
            )
        )
        if tools is not None and tool_material:
            raise ValueError("tools and direct_tools are mutually exclusive.")
        resolved_invocation_loop_policies = tuple(invocation_loop_policies)
        profile_app = CayuApp(enable_logging=False)
        profile_provider = (
            ScriptedModelProvider([], name=provider_name) if provider is None else provider
        )
        if profile_provider.name != provider_name:
            raise ValueError("provider_name must match the supplied provider.")
        profile_app.register_provider(profile_provider, default=True)
        profile_app.register_agent(
            AgentSpec(name="assistant", model=model),
            tools=resolved_tools,
            tool_policy=tool_policy,
            tool_exposure_policy=tool_exposure_policy,
            runtime_hooks=runtime_hooks,
        )
        resolved_profile = execution_profile_admission.resolve_execution_profile_identity(
            registered_agent=profile_app._agents["assistant"],
            provider_name=provider_name,
            model=model,
            durable_system_prompt=durable_system_prompt,
            runtime_name="cayu",
            runtime_version=runtime_version,
            runtime_build_provenance=current_runtime_build_provenance(),
            redactor=profile_app._secret_redactor,
            process_identity=profile_app._execution_profile_process_identity,
            registered_provider=profile_app._providers[provider_name],
            thinking=None if thinking is None else thinking_config_payload(thinking),
            request_budget_limit_ids=(
                ()
                if not request_budget_limits
                else tuple(
                    limit.budget_limit_id
                    for limit in request_budget_limits_for_session(
                        limits=request_budget_limits,
                        agent_name=agent_name,
                        causal_budget_id=causal_budget_id,
                    )
                )
            ),
            structured_output=session_engine_module._execution_profile_structured_output(
                structured_output
            ),
            finalization=execution_profile_admission.model_finalization_material(
                max_steps=max_steps,
                limits=copy_run_limits(limits),
                retry_policy=copy_retry_policy(retry_policy),
            ),
            invocation_loop_policies=resolved_invocation_loop_policies,
            invocation_loop_policy_identities=tuple(
                copy_execution_profile_behavior_identity(policy.execution_profile_identity)
                for policy in resolved_invocation_loop_policies
            ),
        )
    return SessionIdentity(
        provider_name=provider_name,
        model=model,
        runtime_name="cayu",
        runtime_version=runtime_version,
        execution_profile=resolved_profile,
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


async def rebind_test_invocation(
    store: SessionStore,
    session_id: str,
    *,
    target_status: SessionStatus = SessionStatus.RUNNING,
    checkpoint_transform: Callable[
        [Session, dict[str, Any] | None],
        dict[str, Any] | None,
    ] = checkpoint_with_rebound_test_invocation_profile,
) -> Session:
    """Rebind a staged test invocation through the production command boundary."""

    session = await store.load(session_id)
    checkpoint = await store.load_checkpoint(session_id)
    if session is None:
        raise AssertionError(f"Session not found: {session_id}")
    command = prepare_rebind_invocation_command(
        session,
        checkpoint,
        expected_statuses={session.status},
        target_status=target_status,
        checkpoint_transform=checkpoint_transform,
    )
    result = await runtime_checkpoint_session_store(store).apply_invocation_lifecycle_command(
        command
    )
    if type(result) is not InvocationMutationResult:
        raise AssertionError("Test invocation rebind returned an invalid result.")
    return result.session


async def interrupt_and_release_test_invocation(
    store: SessionStore,
    session_id: str,
) -> Session:
    """Model process loss while preserving the invocation's open interaction."""

    session = await store.load(session_id)
    checkpoint = await store.load_checkpoint(session_id)
    if session is None:
        raise AssertionError(f"Session not found: {session_id}")
    active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
    if active_profile is None:
        raise AssertionError("Test invocation has no active lifecycle authority.")
    event = event_with_runtime_envelope_authority(
        Event(
            type=EventType.SESSION_INTERRUPTED,
            session_id=session.id,
            agent_name=session.agent_name,
            environment_name=session.environment_name,
            payload={"reason": "test_process_loss"},
        ),
        "session_id",
    )
    interrupted = await store.transition_status(
        session.id,
        from_statuses={session.status},
        to_status=SessionStatus.INTERRUPTED,
    )
    await store.append_event(session.id, event)
    runtime_store = runtime_checkpoint_session_store(store)
    released = await runtime_store.apply_invocation_lifecycle_command(
        _release_invocation_command_with_cleanup_authority(
            ReleaseInvocationCommand(
                session_id=interrupted.id,
                expected_session_instance_id=interrupted.instance_id,
                expected_run_epoch=active_profile.run_epoch,
                expected_active_profile=active_profile,
                terminal_session_event=event,
            )
        )
    )
    if type(released) is not InvocationReleaseResult:
        raise AssertionError("Test invocation release returned an invalid result.")
    return released.session


async def admit_test_invocation(
    store: SessionStore,
    session_id: str,
    *,
    interaction_started_event: Event,
    interaction_source_messages: Iterable[Message] = (),
    target_profile: ExecutionProfileIdentity | None = None,
    execution_profile_decision: ExecutionProfileDecision | None = None,
    model_transition: SessionModelTransition | None = None,
    checkpoint_transform: Callable[
        [Session, dict[str, Any] | None],
        dict[str, Any] | None,
    ]
    | None = None,
) -> Session:
    """Admit a new test interaction through the production command boundary."""

    session = await store.load(session_id)
    checkpoint = await store.load_checkpoint(session_id)
    if session is None:
        raise AssertionError(f"Session not found: {session_id}")
    active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
    interaction_id = interaction_started_event.interaction_id
    if interaction_id is None:
        raise AssertionError("Test invocation admission requires an interaction id.")
    profile = target_profile
    if profile is None:
        profile = (
            execution_profile_from_session_metadata(session.metadata)
            if active_profile is None
            else active_profile.profile
        )
    desired_checkpoint = checkpoint
    if checkpoint_transform is not None:
        desired_checkpoint = checkpoint_transform(session, checkpoint)
    if desired_checkpoint is not None:
        desired_checkpoint = dict(desired_checkpoint)
        for authority_key in (
            CHECKPOINT_SCHEMA_VERSION_KEY,
            ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
            INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
        ):
            if checkpoint is not None and authority_key in checkpoint:
                desired_checkpoint[authority_key] = checkpoint[authority_key]
            else:
                desired_checkpoint.pop(authority_key, None)
    command = AdmitInvocationCommand(
        session_id=session.id,
        expected_session_instance_id=session.instance_id,
        expected_statuses=(session.status,),
        expected_run_epoch=session.run_epoch,
        expected_checkpoint_sha256=invocation_checkpoint_state_sha256(checkpoint),
        target_active_profile=ActiveInvocationExecutionProfile(
            session_id=session.id,
            interaction_id=interaction_id,
            run_epoch=session.run_epoch + 1,
            profile=profile,
        ),
        checkpoint_patch=InvocationCheckpointPatch(
            mutation=runtime_publication_checkpoint_mutation(
                checkpoint,
                desired_checkpoint,
            )
        ),
        interaction_source_messages=tuple(interaction_source_messages),
        tool_capability_ceiling=tool_capability_ceiling_from_session_metadata(session.metadata),
        interaction_started_event=interaction_started_event,
        execution_profile_decision=execution_profile_decision,
        model_transition=model_transition,
        expected_active_profile=active_profile,
    )
    result = await runtime_checkpoint_session_store(store).apply_invocation_lifecycle_command(
        command
    )
    if type(result) is not InvocationMutationResult:
        raise AssertionError("Test invocation admission returned an invalid result.")
    return result.session
