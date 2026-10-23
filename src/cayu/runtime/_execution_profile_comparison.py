"""Read-only comparison of one stored session profile with the current registration.

The candidate is built the way Runtime builds it at the session's next boundary:
through the continuation plan at a continuation boundary, and through the
ordinary resume profile builder at a resume boundary. Per-request settings come
from the session's recorded model-step semantics. Classes that depend on values
only the next caller supplies are reported as not compared, never as changed,
unless the stored identity is opaque: no new process can reproduce it, so the
class stays changed and the prediction is the rejection Runtime would record.
A run whose process stopped with nothing pending is compared at the resume
boundary: a continuation first recovers that abandoned run, which closes its
interaction, and then admits the new request as an ordinary resume.
Registrations are looked up through the same registries Runtime uses, so a
missing agent, provider or environment raises the error a continuation would.
Nothing here writes to the store, claims the session, or calls application code
beyond the identity hooks that ordinary profile resolution already reads.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from cayu.budgets.base import BudgetPolicy, copy_budget_policy
from cayu.budgets.run_limits import RunLimits
from cayu.build_provenance import current_runtime_build_provenance
from cayu.execution_profiles import (
    ExecutionProfileAdmissionBoundary,
    ExecutionProfileComponentClass,
    ExecutionProfileIdentity,
    ExecutionProfileIdentityStrength,
    execution_profile_with_component,
)
from cayu.messages import Message
from cayu.providers.retry_policy import RetryPolicy
from cayu.runtime import _execution_profile_admission as execution_profile_admission
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _session_execution_profile
from cayu.runtime._execution_profile_continuation import (
    ExecutionProfileContinuation,
    _runtime_version,
)
from cayu.runtime._recovery_coordinator import recorded_pending_invocation_semantics
from cayu.runtime._resume_configuration import (
    inherit_resume_configuration,
    recorded_admitted_configuration,
    recorded_invocation_semantics,
)
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.runtime.loop_policies import LoopPolicy
from cayu.sessions._execution_profile_checkpoint import (
    _CONTINUATION_CHECKPOINT_KEYS,
    execution_profile_from_session_metadata,
    session_execution_profiles,
)
from cayu.sessions._model_failover import ModelFailoverPolicy, ModelTarget
from cayu.sessions.records import Session, SessionRuntimeIdentity, SessionStatus
from cayu.sessions.requests import DEFAULT_MAX_STEPS, ResumeRequest
from cayu.sessions.store import SessionStore
from cayu.tools.exposure import resolve_tool_capability_ceiling
from cayu.vaults.redaction import SecretRedactor

_T = TypeVar("_T")

# Settings a caller passes per request; Runtime records them with pending work.
_RECORDED_REQUEST_CLASSES = (
    ExecutionProfileComponentClass.PROVIDER_REQUEST_POLICY,
    ExecutionProfileComponentClass.INVOCATION_BUDGET_POLICY,
    ExecutionProfileComponentClass.STRUCTURED_OUTPUT,
    ExecutionProfileComponentClass.FINALIZATION,
)
# Of those, the classes a finished interaction keeps no request values for.
_RESUME_REQUEST_CLASSES = _RECORDED_REQUEST_CLASSES[:3]
# A continuation or resume without request loop_policies passes an empty tuple.
_NO_REQUEST_LOOP_POLICIES: tuple[LoopPolicy, ...] = ()
_LOOP_POLICIES_REASON = (
    "The session was run with request loop_policies. They are passed again by the call "
    "that continues the session and are not stored, so this command cannot compare them."
)
_NO_PENDING_RECORD_REASON = (
    "The pending work is not an approval, user input or tool round, so this command cannot "
    "reconstruct the run settings recovery restores for it."
)
_RESUME_REQUEST_REASON = (
    "Each ResumeRequest sets thinking, budget_limits, structured_output and failover, and a "
    "finished interaction does not store them, so this command cannot tell a request "
    "setting from a code change here. Pass the values the session was run with; this "
    "class also covers the agent's provider_options and thinking defaults."
)
_NO_FINALIZATION_REASON = (
    "The session has no recorded run controls for its latest model step, so max_steps, "
    "limits and retry policy cannot be reconstructed."
)
_ABANDONED_RUN_NOTE = (
    "The session's process stopped while it was running and nothing is pending. Resuming "
    "it first recovers that run, which closes its interaction, so the new request is "
    "compared at the resume boundary."
)
# Statuses whose lost owner a continuation recovers before it admits new work.
_ABANDONED_RUN_STATUSES = frozenset({SessionStatus.RUNNING, SessionStatus.INTERRUPTING})


@dataclass(frozen=True, slots=True)
class SessionExecutionProfileComparison:
    """Stored and current profiles at the session's next boundary."""

    session: Session
    boundary: ExecutionProfileAdmissionBoundary
    stored_profile: ExecutionProfileIdentity
    candidate_profile: ExecutionProfileIdentity
    not_compared: tuple[tuple[ExecutionProfileComponentClass, str], ...]
    notes: tuple[str, ...] = ()


class SessionExecutionProfileUnavailable(RuntimeError):
    """The session has no stored profile that its next boundary would compare."""


class ComparedSessionNotFound(LookupError):
    """The session store has no session with this id."""


class SessionRegistrationMissing(LookupError):
    """A registration the session's next boundary resolves is not in this application.

    ``message`` is the registry's own error, the one Runtime raises when it
    continues or resumes the session.
    """

    def __init__(self, *, kind: str, name: str, message: str) -> None:
        self.kind = kind
        self.name = name
        self.message = message
        super().__init__(message)


def _registration(kind: str, lookup: Callable[[str], _T], name: str) -> _T:
    try:
        return lookup(name)
    except KeyError as exc:
        message = str(exc.args[0]) if exc.args else f"{kind.capitalize()} not registered: {name}"
        raise SessionRegistrationMissing(kind=kind, name=name, message=message) from exc


async def compare_session_execution_profile(
    *,
    store: SessionStore,
    session: Session,
    continuation: ExecutionProfileContinuation,
    get_registered_agent: Callable[[str], runtime_records.RegisteredAgentState],
    get_registered_provider: Callable[[str], runtime_records.RegisteredProvider],
    get_registered_environment: Callable[[str], runtime_records.RegisteredEnvironment | None],
    runtime_hooks: tuple[runtime_records.RegisteredRuntimeHook, ...],
    loop_policies: tuple[LoopPolicy, ...],
    loop_policy_identities: tuple[ExecutionProfileBehaviorIdentity | None, ...],
    budget_policy: BudgetPolicy | None,
    effective_retry_policy: Callable[[RetryPolicy | None], RetryPolicy],
    process_identity: str,
    redactor: SecretRedactor,
) -> SessionExecutionProfileComparison:
    """Compare one session's stored profile with this registration's candidate.

    Raises ``SessionRegistrationMissing`` when the session's agent, provider,
    environment or a failover provider is not registered, as its continuation
    or resume would fail, and ``SessionExecutionProfileUnavailable`` when the
    session has no comparable stored profile.
    """

    # Runtime resolves the agent, then the provider, then the environment.
    registered_agent = _registration("agent", get_registered_agent, session.agent_name)
    registered_provider = _registration("provider", get_registered_provider, session.provider_name)
    registered_environment = (
        None
        if session.environment_name is None
        else _registration("environment", get_registered_environment, session.environment_name)
    )
    checkpoint = await store.load_checkpoint(session.id)
    stage = await store.load_active_model_completion_stage(session.id)
    stored = session_execution_profiles(
        session, checkpoint, model_completion_pending=stage is not None
    )
    next_profile = (
        stored.active_invocation.profile
        if stored.boundary is ExecutionProfileAdmissionBoundary.CONTINUATION
        and stored.active_invocation is not None
        else stored.expected
    )
    if next_profile is not None and next_profile.model_failover is not None:
        for entry in next_profile.model_failover.plan.candidates:
            _registration("provider", get_registered_provider, entry.provider_name)
    not_compared: list[tuple[ExecutionProfileComponentClass, str]] = []
    notes: list[str] = []
    boundary = stored.boundary
    semantics = None
    if boundary is ExecutionProfileAdmissionBoundary.CONTINUATION:
        # Restore the pending record's settings as a continuation without
        # overrides does.
        semantics = recorded_pending_invocation_semantics(
            checkpoint,
            session=session,
            redactor=redactor,
            effective_retry_policy=effective_retry_policy,
        )
        if (
            semantics is None
            and stage is None
            and stored.expected is not None
            and _abandoned_run_without_pending_work(session, checkpoint)
        ):
            # Runtime's resume recovers the abandoned run first, which releases
            # its invocation, and then admits at the ordinary resume boundary.
            boundary = ExecutionProfileAdmissionBoundary.RESUME
            notes.append(_ABANDONED_RUN_NOTE)
    if boundary is ExecutionProfileAdmissionBoundary.CONTINUATION:
        # Reuse Runtime's continuation reconstruction. Without semantics it
        # keeps the recorded request classes.
        plan = await continuation.inspect_plan(
            session=session,
            checkpoint=checkpoint,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            request_loop_policies=_NO_REQUEST_LOOP_POLICIES,
            budget_policy=copy_budget_policy(budget_policy),
            invocation_semantics_available=semantics is not None,
            request_budget_limits=() if semantics is None else semantics.budget_limits,
            structured_output=None if semantics is None else semantics.structured_output,
            thinking=None if semantics is None else semantics.thinking,
            max_steps=None if semantics is None else semantics.max_steps,
            limits=None if semantics is None else semantics.limits,
            retry_policy=None if semantics is None else semantics.retry_policy,
        )
        stored_profile = plan.snapshot.profile
        candidate = plan.candidate_profile
        if semantics is None and stored_profile.schema_version >= 4:
            candidate = await _rebuild_opaque_recorded_classes(
                candidate,
                stored_profile,
                store=store,
                session=session,
                checkpoint=checkpoint,
                continuation=continuation,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                budget_policy=budget_policy,
                effective_retry_policy=effective_retry_policy,
                not_compared=not_compared,
            )
    elif boundary is ExecutionProfileAdmissionBoundary.RESUME and stored.expected is not None:
        stored_profile = execution_profile_from_session_metadata(session.metadata)
        context = await recorded_invocation_semantics(store, session, checkpoint)
        configuration = recorded_admitted_configuration(checkpoint, session)
        request, _ = await inherit_resume_configuration(
            store,
            session,
            ResumeRequest(
                session_id=session.id, messages=[Message.text("user", "Compare execution profile.")]
            ),
            effective_retry_policy(None),
        )
        candidate = _resume_candidate(
            session=session,
            stored_profile=stored_profile,
            request=request,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            registered_environment=registered_environment,
            get_registered_provider=get_registered_provider,
            runtime_hooks=runtime_hooks,
            loop_policies=loop_policies,
            loop_policy_identities=loop_policy_identities,
            budget_policy=budget_policy,
            effective_retry_policy=effective_retry_policy,
            process_identity=process_identity,
            redactor=redactor,
        )
        unknown = [
            (component_class, _RESUME_REQUEST_REASON)
            for component_class in _RESUME_REQUEST_CLASSES
            if component_class is not ExecutionProfileComponentClass.STRUCTURED_OUTPUT
            or (context is None and configuration is None)
        ]
        if context is None and configuration is None:
            unknown.append((ExecutionProfileComponentClass.FINALIZATION, _NO_FINALIZATION_REASON))
        if stored_profile.schema_version >= 4:
            for component_class, reason in unknown:
                candidate = _assume_caller_repeats(
                    candidate, stored_profile, component_class, reason, not_compared
                )
    else:
        issues = ", ".join(stored.issues) or "the session has no recorded execution profile"
        raise SessionExecutionProfileUnavailable(
            f"The session's stored execution profile cannot be compared: {issues}."
        )
    if stored_profile.schema_version >= 2:
        candidate = _assume_caller_repeats(
            candidate,
            stored_profile,
            ExecutionProfileComponentClass.INVOCATION_POLICIES,
            _LOOP_POLICIES_REASON,
            not_compared,
        )
    return SessionExecutionProfileComparison(
        session=session,
        boundary=boundary,
        stored_profile=stored_profile,
        candidate_profile=candidate,
        not_compared=tuple(not_compared),
        notes=tuple(notes),
    )


def _abandoned_run_without_pending_work(
    session: Session, checkpoint: dict[str, Any] | None
) -> bool:
    """Whether the session's next continuation only closes a run whose process stopped.

    Such a session is still running, but no approval, user input, tool round,
    foreground handoff or provider-operation resolution is waiting, so the
    abandoned-execution recovery a continuation runs first interrupts the run
    and releases its invocation instead of continuing it.
    """

    return session.status in _ABANDONED_RUN_STATUSES and not any(
        checkpoint is not None and checkpoint.get(key) is not None
        for key in _CONTINUATION_CHECKPOINT_KEYS
    )


async def _rebuild_opaque_recorded_classes(
    candidate: ExecutionProfileIdentity,
    stored_profile: ExecutionProfileIdentity,
    *,
    store: SessionStore,
    session: Session,
    checkpoint: dict[str, Any] | None,
    continuation: ExecutionProfileContinuation,
    registered_agent: runtime_records.RegisteredAgentState,
    registered_provider: runtime_records.RegisteredProvider,
    budget_policy: BudgetPolicy | None,
    effective_retry_policy: Callable[[RetryPolicy | None], RetryPolicy],
    not_compared: list[tuple[ExecutionProfileComponentClass, str]],
) -> ExecutionProfileIdentity:
    """Keep recorded request classes out of the prediction unless they are opaque.

    Runtime continues such a session with the settings of the next request, so
    every recorded class is rebuilt in the continuing process. An opaque one is
    never reproduced there; it is rebuilt here the way a plain ResumeRequest
    rebuilds it, so the class is reported as changed instead of copied.
    """

    opaque = [
        component_class
        for component_class in _RECORDED_REQUEST_CLASSES
        if stored_profile.component(component_class).strength
        is ExecutionProfileIdentityStrength.PROCESS_LOCAL
    ]
    rebuilt = None
    if opaque:
        context = await recorded_invocation_semantics(store, session, checkpoint)
        rebuilt = (
            await continuation.inspect_plan(
                session=session,
                checkpoint=checkpoint,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                request_loop_policies=_NO_REQUEST_LOOP_POLICIES,
                budget_policy=copy_budget_policy(budget_policy),
                invocation_semantics_available=True,
                max_steps=DEFAULT_MAX_STEPS if context is None else context.max_steps,
                limits=RunLimits() if context is None else context.limits,
                retry_policy=effective_retry_policy(
                    None if context is None else context.retry_policy
                ),
            )
        ).candidate_profile
    for component_class in _RECORDED_REQUEST_CLASSES:
        if rebuilt is not None and component_class in opaque:
            candidate = execution_profile_with_component(
                candidate, rebuilt.component(component_class)
            )
        else:
            not_compared.append((component_class, _NO_PENDING_RECORD_REASON))
    return candidate


def _assume_caller_repeats(
    candidate: ExecutionProfileIdentity,
    stored_profile: ExecutionProfileIdentity,
    component_class: ExecutionProfileComponentClass,
    reason: str,
    not_compared: list[tuple[ExecutionProfileComponentClass, str]],
) -> ExecutionProfileIdentity:
    """Keep a caller-supplied class out of the prediction when the caller can repeat it.

    An opaque stored identity is never copied: no value a later caller passes,
    in this or any new process, reproduces it, so Runtime rejects the class and
    the comparison reports it as changed.
    """

    recorded = stored_profile.component(component_class)
    if candidate.component(component_class) == recorded:
        return candidate
    if recorded.strength is ExecutionProfileIdentityStrength.PROCESS_LOCAL:
        return candidate
    not_compared.append((component_class, reason))
    return execution_profile_with_component(candidate, recorded)


def _resume_candidate(
    *,
    session: Session,
    stored_profile: ExecutionProfileIdentity,
    request: ResumeRequest,
    registered_agent: runtime_records.RegisteredAgentState,
    registered_provider: runtime_records.RegisteredProvider,
    registered_environment: runtime_records.RegisteredEnvironment | None,
    get_registered_provider: Callable[[str], runtime_records.RegisteredProvider],
    runtime_hooks: tuple[runtime_records.RegisteredRuntimeHook, ...],
    loop_policies: tuple[LoopPolicy, ...],
    loop_policy_identities: tuple[ExecutionProfileBehaviorIdentity | None, ...],
    budget_policy: BudgetPolicy | None,
    effective_retry_policy: Callable[[RetryPolicy | None], RetryPolicy],
    process_identity: str,
    redactor: SecretRedactor,
) -> ExecutionProfileIdentity:
    """Mirror the resume builder for a plain ResumeRequest after inherited run controls."""

    runtime_identity = SessionRuntimeIdentity(
        runtime_name="cayu",
        runtime_version=_runtime_version(),
        runtime_build_provenance=current_runtime_build_provenance(),
    )
    ceiling = resolve_tool_capability_ceiling(
        None,
        registered_agent.tool_capabilities,
        maximum=_session_execution_profile._session_tool_capability_ceiling(session),
    )
    budget_policy = copy_budget_policy(budget_policy)

    def build(provider: runtime_records.RegisteredProvider, model: str) -> ExecutionProfileIdentity:
        candidate = _session_execution_profile._execution_profile_identity(
            registered_agent=registered_agent,
            provider_name=provider.name,
            registered_provider=provider,
            model=model,
            durable_system_prompt=None,
            redactor=redactor,
            registered_environment=registered_environment,
            process_identity=process_identity,
            runtime_hooks=runtime_hooks,
            loop_policies=loop_policies,
            loop_policy_execution_profile_identities=loop_policy_identities,
            budget_policy=budget_policy,
            request_budget_limits=(),
            causal_budget_id=session.causal_budget_id,
            structured_output=request.structured_output,
            thinking=None,
            max_steps=request.max_steps,
            limits=request.limits,
            retry_policy=effective_retry_policy(request.retry_policy),
            tool_capability_ceiling=ceiling,
            runtime_identity=runtime_identity,
            tool_completion=request.tool_completion,
        )
        return execution_profile_with_component(
            candidate,
            stored_profile.component(ExecutionProfileComponentClass.DURABLE_SYSTEM_PROJECTION),
        )

    stored_failover = stored_profile.model_failover
    if stored_failover is None:
        return build(registered_provider, session.model)
    policy = ModelFailoverPolicy(
        fallbacks=tuple(
            ModelTarget(provider_name=entry.provider_name, model=entry.model)
            for entry in stored_failover.plan.candidates[1:]
        ),
        max_total_attempts=stored_failover.plan.max_total_attempts,
    )
    return execution_profile_admission.resolve_model_failover_execution_profile(
        policy=policy,
        primary=ModelTarget(provider_name=registered_provider.name, model=session.model),
        registered_primary=registered_provider,
        resolve_provider=get_registered_provider,
        resolve_candidate_profile=build,
        redactor=redactor,
    ).profile
