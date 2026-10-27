"""Shared initial-session preparation below ordinary and governed execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any
from uuid import uuid4

from cayu._validation import (
    copy_json_value,
    require_clean_nonblank,
)
from cayu.budgets.base import (
    BudgetPolicy,
    copy_budget_policy,
)
from cayu.build_provenance import current_runtime_build_provenance
from cayu.context.structured_output import StructuredOutputSpec
from cayu.context.structured_output import (
    _require_native_structured_output_support as _require_provider_native_output_support,
)
from cayu.deadlines import (
    ExecutionDeadline,
    current_execution_deadline,
    effective_deadline,
)
from cayu.environments.base import NO_ENVIRONMENT, WorkspaceInstructions
from cayu.execution_profiles import (
    ExecutionProfileIdentity,
    changed_execution_profile_components,
    unavailable_execution_profile_components,
)
from cayu.messages import Message
from cayu.providers.retry_policy import RetryPolicy
from cayu.runtime import _execution_profile_admission as execution_profile_admission
from cayu.runtime import _execution_profile_continuation as execution_profile_continuation
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _session_execution_profile
from cayu.runtime import _session_request_boundary as session_request_boundary
from cayu.runtime._durable_subagents import durable_subagent_authority_rejected
from cayu.runtime._environment_lifecycle import (
    EnvironmentLifecycle,
    render_initial_system_prompt_with_contributions,
)
from cayu.runtime._execution_profile_continuation import ExecutionProfileContinuation
from cayu.runtime._model_policy import PolicySelection
from cayu.runtime._model_step_executor import (
    _model_request_messages,
    _model_request_tools,
    _tool_capability_ceiling_exposure,
    preflight_model_thinking,
    preflight_portable_model_material,
)
from cayu.runtime._session_finalization import SessionFinalization
from cayu.runtime._task_execution_admission import (
    require_unretired_task_session_identity,
    verifier_aware_task_execution_outcome,
)
from cayu.runtime._task_store_operation_boundary import (
    capture_task_store_operation,
    raise_task_store_operation_failure,
)
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.runtime.execution_profiles import ExecutionProfileMismatchError
from cayu.runtime.loop_policies import LoopPolicy
from cayu.sessions._model_failover import ModelTarget
from cayu.sessions.invocation import SessionExecutionSource
from cayu.sessions.records import Session, SessionIdentity
from cayu.sessions.requests import (
    RunRequest,
    copy_run_request,
    run_request_authority_is_runtime_generated,
    run_request_with_task_invocation,
    runtime_prepared_session_authority,
)
from cayu.sessions.store import SessionStore
from cayu.tasks.admission import (
    WorkAttemptAdmission,
    WorkAttemptAdmissionState,
    WorkAttemptRecoveryRequired,
)
from cayu.tasks.store import TaskStore
from cayu.tools.exposure import (
    ToolCapabilityCeiling,
    resolve_tool_capability_ceiling,
)
from cayu.tools.gateway import require_targeted_tool_gateway_capacity
from cayu.tools.grants import (
    PreparedTargetedToolGrant,
    TargetedToolGrant,
    prepare_targeted_tool_grants,
)
from cayu.tools.targeted_projection import (
    TargetedToolProjectionKind,
    resolve_targeted_tool_projection,
)
from cayu.vaults.redaction import SecretRedactor


def _require_native_structured_output_support(
    structured_output: StructuredOutputSpec | None,
    *,
    registered_provider: runtime_records.RegisteredProvider,
) -> None:
    """Reject an unusable ``strategy=NATIVE`` spec at an entry point.

    Called by every entrance before it creates a session or transitions its
    status, so the typed error reaches the caller with no persisted state
    changed (model-pattern routing can select the provider by model name
    alone). Raises ``NativeStructuredOutputUnsupported`` when the resolved
    provider has no native mode, then lets the provider's own schema preflight
    reject schemas its native mode would refuse at request time
    (``NativeStructuredOutputSchemaInvalid``). One pre-existing mid-run call
    in ``_run_session`` reuses this helper as a typed backstop for missed
    entrances; a raise there fails the already-running session cleanly.
    """
    _require_provider_native_output_support(
        structured_output,
        provider_name=registered_provider.name,
        provider=registered_provider.provider,
    )


@dataclass(frozen=True, slots=True)
class _PreparedInitialRun:
    """Validated new-session material frozen before the store creation boundary."""

    request: RunRequest
    registered_agent: runtime_records.RegisteredAgentState
    registered_provider: runtime_records.RegisteredProvider
    registered_environment: runtime_records.RegisteredEnvironment | None
    workspace_instructions: WorkspaceInstructions | None
    rendered_system_prompt: str | None
    prompt_contributions: tuple[Any, ...]
    execution_profile: ExecutionProfileIdentity
    model_failover: execution_profile_admission.ModelFailoverProfileResolution | None
    tool_capability_ceiling: ToolCapabilityCeiling
    targeted_tool_grants: tuple[PreparedTargetedToolGrant, ...]
    budget_policy: BudgetPolicy | None
    session_identity: SessionIdentity
    policy_evidence: bytes | None
    # The Cayu Cloud settings revision and values this session starts with.
    managed_settings_evidence: dict[str, Any] | None = None


def _session_identity(
    *,
    provider_name: str,
    model: str,
    execution_profile: ExecutionProfileIdentity | None = None,
) -> SessionIdentity:
    return SessionIdentity(
        provider_name=provider_name,
        model=model,
        runtime_name="cayu",
        runtime_version=execution_profile_continuation._runtime_version(),
        runtime_build_provenance=current_runtime_build_provenance(),
        execution_profile=execution_profile,
    )


def _inherits_no_environment(request: RunRequest, parent: Session | None) -> bool:
    """Return whether a subagent child inherits its parent's lack of an environment.

    A built-in subagent request carries its parent's recorded environment, or
    its ``SubagentSpec`` override. Without either, a parent stored with no
    environment means the child runs without one too, not with a default.
    """

    return (
        request._runtime_invocation_source is SessionExecutionSource.SUBAGENT
        and request.environment_name is None
        and parent is not None
        and parent.environment_name is None
    )


def _with_environment_name(request: RunRequest, environment_name: str) -> RunRequest:
    # ``copy_run_request`` deliberately preserves the request's private runtime
    # authority provenance. Reconstructing a fresh model here would silently
    # downgrade trusted subagent/session lineage back to caller input.
    return copy_run_request(request).model_copy(update={"environment_name": environment_name})


class SessionPreparation:
    """Freeze session authority and runtime collaborators before admission."""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        task_store: TaskStore | None,
        environment_lifecycle: EnvironmentLifecycle,
        execution_profile_continuation: ExecutionProfileContinuation,
        session_finalization: SessionFinalization,
        secret_redactor: SecretRedactor,
        runtime_hooks: tuple[runtime_records.RegisteredRuntimeHook, ...],
        loop_policies: tuple[LoopPolicy, ...],
        loop_policy_execution_profile_identities: tuple[
            ExecutionProfileBehaviorIdentity | None, ...
        ],
        get_budget_policy: Callable[[], BudgetPolicy | None],
        get_registered_agent: Callable[[str], runtime_records.RegisteredAgentState],
        get_registered_provider: Callable[[str | None], runtime_records.RegisteredProvider],
        route_registered_provider_for_model: Callable[
            [str], runtime_records.RegisteredProvider | None
        ],
        get_registered_environment: Callable[
            [str | None], runtime_records.RegisteredEnvironment | None
        ],
        get_registered_environment_for_session: Callable[
            [str | None], runtime_records.RegisteredEnvironment | None
        ],
        effective_retry_policy: Callable[[RetryPolicy | None], RetryPolicy],
        execution_profile_process_identity: str,
        policy_selection: Callable[[str], PolicySelection | None] | None = None,
        managed_model: Callable[[str], tuple[bool, str | None]] | None = None,
        managed_evidence: Callable[[str, dict[str, Any]], dict[str, Any] | None] | None = None,
    ) -> None:
        self.session_store = session_store
        self.task_store = task_store
        self._environment_lifecycle = environment_lifecycle
        self._execution_profile_continuation = execution_profile_continuation
        self._session_finalization = session_finalization
        self._secret_redactor = secret_redactor
        self._runtime_hooks = runtime_hooks
        self._loop_policies = loop_policies
        if len(loop_policies) != len(loop_policy_execution_profile_identities):
            raise ValueError("Loop-policy execution-profile identities are inconsistent.")
        self._loop_policy_execution_profile_identities = loop_policy_execution_profile_identities
        self._get_budget_policy = get_budget_policy
        self._get_registered_agent = get_registered_agent

        def resolve_provider(name: str | None = None) -> runtime_records.RegisteredProvider:
            return get_registered_provider(name)

        self._get_registered_provider = resolve_provider

        def route_provider_for_model(*, model: str) -> runtime_records.RegisteredProvider | None:
            return route_registered_provider_for_model(model)

        self._route_registered_provider_for_model = route_provider_for_model
        self._get_registered_environment = get_registered_environment
        self._get_registered_environment_for_session = get_registered_environment_for_session
        self._effective_retry_policy = effective_retry_policy
        self._execution_profile_process_identity = require_clean_nonblank(
            execution_profile_process_identity, "execution_profile_process_identity"
        )
        self._policy_selection = policy_selection
        self._managed_model = managed_model
        self._managed_evidence = managed_evidence

    def resolve_new_run_environment(
        self,
        environment_name: str | None,
        *,
        registered_agent: runtime_records.RegisteredAgentState,
    ) -> runtime_records.RegisteredEnvironment | None:
        """Resolve a new run's environment.

        The request's ``environment_name`` wins, then the agent's registered
        default (``NO_ENVIRONMENT`` selects none), then the application default.
        """

        if environment_name is None:
            agent_default = registered_agent.default_environment_name
            if agent_default is NO_ENVIRONMENT:
                return None
            environment_name = agent_default
        return self._get_registered_environment(environment_name)

    def manages_model(self, agent_name: str) -> bool:
        """Whether Cayu Cloud's managed settings choose this agent's model."""

        return self._managed_model is not None and self._managed_model(agent_name)[0]

    def _resolve_initial_model_target(
        self,
        request: RunRequest,
        registered_agent: runtime_records.RegisteredAgentState,
        *,
        select_current_policy: bool = True,
    ) -> tuple[runtime_records.RegisteredProvider, str, bytes | None]:
        prepared = runtime_prepared_session_authority(request)
        managed, managed_model = (
            (False, None)
            if self._managed_model is None or not select_current_policy
            else self._managed_model(request.agent_name)
        )
        # A queued policy selection belongs to the admitted child, not the
        # worker's current default. Non-policy children still undergo ordinary
        # agent/profile compatibility checks and cannot adopt policy retroactively.
        # Likewise a child whose model Cayu Cloud manages keeps the model it was
        # prepared with, so a later change in Cloud doesn't break its profile.
        if prepared is not None and (prepared.policy_evidence is not None or managed):
            if request.target is not None and (
                request.target.provider_name != prepared.provider_name
                or request.target.model != prepared.model
            ):
                raise durable_subagent_authority_rejected()
            return (
                self._get_registered_provider(prepared.provider_name),
                prepared.model,
                prepared.policy_evidence,
            )
        if request.target is not None:
            return (
                self._get_registered_provider(request.target.provider_name),
                request.target.model,
                None,
            )
        selection = (
            None
            if self._policy_selection is None or prepared is not None or not select_current_policy
            else self._policy_selection(request.agent_name)
        )
        if selection is not None:
            return (
                self._get_registered_provider(selection.target.provider_name),
                selection.target.model,
                selection.evidence,
            )
        # Then Cayu Cloud's value for new sessions, then the agent's own model.
        model = registered_agent.spec.model if managed_model is None else managed_model
        if registered_agent.spec.provider_name is not None:
            return self._get_registered_provider(registered_agent.spec.provider_name), model, None
        return (
            self._route_registered_provider_for_model(model=model)
            or self._get_registered_provider(),
            model,
            None,
        )

    def resolve_initial_model_target(self, request: RunRequest) -> ModelTarget:
        """Resolve initial provider/model routing without preparing or admitting a session."""

        if type(request) is not RunRequest:
            raise TypeError("Initial model-target resolution requires a RunRequest.")
        registered_agent = self._get_registered_agent(request.agent_name)
        registered_provider, model, _ = self._resolve_initial_model_target(
            request,
            registered_agent,
        )
        return ModelTarget(provider_name=registered_provider.name, model=model)

    def _prepare_targeted_tool_grants(
        self,
        grants: tuple[TargetedToolGrant, ...],
        *,
        registered_agent: runtime_records.RegisteredAgentState,
        capability_ceiling: ToolCapabilityCeiling,
    ) -> tuple[PreparedTargetedToolGrant, ...]:
        """Freeze typed grant requests against current registered authority."""

        if not grants:
            return ()
        if registered_agent.targeted_tool_mode is None:
            raise ValueError(
                "Targeted tool grants require targeted_tool_mode at agent registration."
            )
        if (
            not self.session_store.supports_targeted_tool_grants
            or not self.session_store.supports_public_authority_aliases
            or self.session_store.public_authority_alias_codec is None
        ):
            raise RuntimeError(
                "Targeted tool grants require a SessionStore with durable grant state "
                "and an explicitly configured public authority alias keyring."
            )
        return prepare_targeted_tool_grants(
            grants,
            catalogue=registered_agent.tool_catalogue,
            capability_ceiling=capability_ceiling,
        )

    def _require_targeted_tool_delivery_capacity(
        self,
        prepared: tuple[PreparedTargetedToolGrant, ...],
        *,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        model: str,
        model_failover: execution_profile_admission.ModelFailoverProfileResolution | None,
    ) -> None:
        """Size grants for call_tool when any target this run can use delivers through it."""

        if not prepared:
            return
        targets = (
            ((registered_provider, model),)
            if model_failover is None
            else tuple(
                zip(
                    model_failover.registered_providers,
                    (candidate.model for candidate in model_failover.plan.candidates),
                    strict=True,
                )
            )
        )
        if any(
            resolve_targeted_tool_projection(
                registered_agent.targeted_tool_mode,
                provider=candidate_provider.provider,
                model=candidate_model,
            )
            is TargetedToolProjectionKind.CALL_TOOL
            for candidate_provider, candidate_model in targets
        ):
            require_targeted_tool_gateway_capacity(
                prepared,
                catalogue=registered_agent.tool_catalogue,
                redactor=self._secret_redactor,
            )

    async def prepare_initial_run(
        self,
        request: RunRequest,
        *,
        registered_agent_override: runtime_records.RegisteredAgentState | None = None,
        admit_session: bool = True,
        store_resolved_existing_session_id: str | None = None,
        expected_execution_profile: ExecutionProfileIdentity | None = None,
        expected_registered_environment: runtime_records.RegisteredEnvironment | None = None,
        expected_context_policy: object | None = None,
        allow_work_attempt_admission: bool = False,
        prepared_work_attempt: WorkAttemptAdmission | None = None,
        allow_existing_session_id: bool = False,
        retained_policy_selection: PolicySelection | None = None,
        inspect_only: bool = False,
        session_environment: bool = False,
    ) -> _PreparedInitialRun | None:
        """Resolve one new-session request, optionally without ordinary admission.

        ``inspect_only`` resolves a run that is never admitted: it does not load
        the session the request names, and for a generated session id it skips
        the session-scoped contract lookups, since no contract can bind an id
        that did not exist. ``session_environment`` resolves
        ``environment_name`` as a stored session's does, so ``None`` means no
        environment rather than the application default.
        """

        if inspect_only and (admit_session or allow_work_attempt_admission):
            raise ValueError("Inspection cannot admit a session or a work attempt.")
        if expected_execution_profile is not None:
            if type(expected_execution_profile) is not ExecutionProfileIdentity:
                raise TypeError(
                    "expected_execution_profile must be an exact ExecutionProfileIdentity."
                )
            expected_execution_profile = ExecutionProfileIdentity.model_validate(
                expected_execution_profile.model_dump(mode="json")
            )

        # The caller's own fields: Cloud doesn't control what the request sets itself.
        explicit_fields = frozenset(request.model_fields_set)
        request = session_request_boundary.prepare_run_request(
            request,
            redactor=self._secret_redactor,
        )
        if request.failover is not None:
            self.session_store._require_model_failover_stage_protocol()
        parent = (
            await self.session_store.load(request.parent_session_id)
            if request.parent_session_id is not None
            else None
        )
        boundary = effective_deadline(
            current_execution_deadline(),
            request.execution_deadline,
            parent.execution_deadline if parent is not None else ExecutionDeadline(),
        )
        if prepared_work_attempt is not None:
            if (
                admit_session
                or not allow_work_attempt_admission
                or prepared_work_attempt.state is not WorkAttemptAdmissionState.PREPARING
                or prepared_work_attempt.run_semantics is None
                or prepared_work_attempt.run_semantics.deadline.model_dump()
                != boundary.model_dump()
                or prepared_work_attempt.session_id != request.session_id
                or prepared_work_attempt.task_id != request.task_id
            ):
                raise WorkAttemptRecoveryRequired("Prepared admission authority changed.")
        elif admit_session or allow_work_attempt_admission:
            boundary.require_admission("run_preparation")
        request = request.model_copy(update={"execution_deadline": boundary})
        generated_session_id = request.session_id is None
        if generated_session_id:
            request = request.model_copy(update={"session_id": str(uuid4())})
        if request.session_id is None:
            raise AssertionError("Run request session identity was not assigned.")
        contract_lookup_session_id = (
            None if inspect_only and generated_session_id else request.session_id
        )
        task_store = self.task_store
        await require_unretired_task_session_identity(
            request.session_id, task_store=task_store, redactor=self._secret_redactor
        )
        task_id = request.task_id
        if task_id is not None:
            self._session_finalization.require_terminal_interaction_publication_protocol()
        if task_id is not None and task_store is not None:
            source_task_outcome = await capture_task_store_operation(
                lambda: task_store.load_invocation_snapshot(task_id),
                operation_name="Work-attempt task invocation lookup",
                redactor=self._secret_redactor,
            )
            if source_task_outcome.failure is not None:
                raise_task_store_operation_failure(source_task_outcome.failure)
            source_task = source_task_outcome.result
            del source_task_outcome
            if source_task is not None and source_task.session_id in {None, request.session_id}:
                request = run_request_with_task_invocation(
                    request,
                    source_task,
                )
        # Freeze application-budget authority before any await can expose the
        # prepared invocation. The same snapshot must govern both the durable
        # execution profile and the model dispatch that follows it.
        deferred_runtime_task_creation = (
            request.task_id is not None
            and run_request_authority_is_runtime_generated(
                request,
                field_name="task_id",
                value=request.task_id,
            )
        )
        # Reject authority that is already attached to an explicit session
        # before workspace/provider extensions participate in preparation. The
        # final call below still performs the atomic ordinary-execution
        # admission, closing a concurrent contract-attachment race.
        (
            requires_completion_decision,
            admission_failure,
        ) = await verifier_aware_task_execution_outcome(
            task_store=self.task_store,
            redactor=self._secret_redactor,
            task_id=request.task_id,
            session_id=contract_lookup_session_id,
            admit_session=False,
            allow_missing_task=deferred_runtime_task_creation,
        )
        if admission_failure is not None:
            del request
            raise_task_store_operation_failure(admission_failure)
        if requires_completion_decision and not allow_work_attempt_admission:
            return None
        budget_policy = copy_budget_policy(self._get_budget_policy())
        registered_agent = (
            self._get_registered_agent(request.agent_name)
            if registered_agent_override is None
            else registered_agent_override
        )
        if registered_agent.spec.name != request.agent_name:
            raise ValueError("Initial-run agent override conflicts with the request agent.")
        if registered_agent.fork_controller_name is not None:
            raise ValueError("A fork-controller body runs only as an overlay fork.")
        if (
            registered_agent.tool_discovery_mode is not None
            and not self.session_store.supports_atomic_session_operation_initialization
        ):
            raise RuntimeError("Tool discovery requires atomic session operation initialization.")
        if (
            expected_context_policy is not None
            and registered_agent.context_policy is not expected_context_policy
        ):
            raise ValueError("Initial-run context policy changed after private admission.")
        if request.tool_capability_ceiling is None:
            effective_tool_capability_ceiling = ToolCapabilityCeiling(
                tool_names=registered_agent.all_registered_tool_exposure.tool_names
            )
        else:
            effective_tool_capability_ceiling = resolve_tool_capability_ceiling(
                request.tool_capability_ceiling,
                registered_agent.tool_capabilities,
            )
        request = request.model_copy(
            update={"tool_capability_ceiling": effective_tool_capability_ceiling}
        )
        targeted_tool_grants = self._prepare_targeted_tool_grants(
            request.tool_grants,
            registered_agent=registered_agent,
            capability_ceiling=effective_tool_capability_ceiling,
        )
        # An explicit target is exact. Otherwise the agent model and optional
        # provider pin feed the existing routing/default selection.
        if retained_policy_selection is not None:
            if not allow_existing_session_id or request.target is not None:
                raise ValueError("Retained policy selection requires a defaulted existing session.")
            registered_provider = self._get_registered_provider(
                retained_policy_selection.target.provider_name
            )
            model = retained_policy_selection.target.model
            policy_evidence = retained_policy_selection.evidence
        else:
            registered_provider, model, policy_evidence = self._resolve_initial_model_target(
                request,
                registered_agent,
                select_current_policy=not allow_existing_session_id,
            )
        for field_name, value in (
            ("agent_name", registered_agent.spec.name),
            ("provider_name", registered_provider.name),
            ("model", model),
        ):
            session_request_boundary.require_secret_free_session_authority(
                value,
                field_name=field_name,
                redactor=self._secret_redactor,
            )
        registered_provider.provider.preflight_model_target(model=model)
        registered_environment = (
            self._get_registered_environment_for_session(request.environment_name)
            if session_environment or _inherits_no_environment(request, parent)
            else self.resolve_new_run_environment(
                request.environment_name,
                registered_agent=registered_agent,
            )
        )
        if (
            expected_registered_environment is not None
            and registered_environment is not expected_registered_environment
        ):
            raise ValueError("Initial-run environment changed after private admission.")
        if registered_environment is not None:
            session_request_boundary.require_secret_free_session_authority(
                registered_environment.spec.name,
                field_name="environment_name",
                redactor=self._secret_redactor,
            )
        if request.environment_name is None and registered_environment is not None:
            request = _with_environment_name(request, registered_environment.spec.name)
        workspace_instructions = await self._environment_lifecycle.load_workspace_instructions(
            registered_environment,
        )
        (
            rendered_system_prompt,
            prompt_contributions,
        ) = render_initial_system_prompt_with_contributions(
            agent_system_prompt=registered_agent.spec.system_prompt,
            workspace_instructions=workspace_instructions,
        )
        resolve_profile = partial(
            _session_execution_profile._execution_profile_identity,
            registered_agent=registered_agent,
            durable_system_prompt=rendered_system_prompt,
            redactor=self._secret_redactor,
            registered_environment=registered_environment,
            process_identity=self._execution_profile_process_identity,
            runtime_hooks=self._runtime_hooks,
            loop_policies=self._loop_policies,
            loop_policy_execution_profile_identities=(
                self._loop_policy_execution_profile_identities
            ),
            request_loop_policies=request.loop_policies,
            request_loop_policy_instance_identities=(
                self._execution_profile_continuation.request_loop_policy_instance_identities(
                    request.loop_policies
                )
            ),
            budget_policy=budget_policy,
            request_budget_limits=request.budget_limits,
            causal_budget_id=(request.causal_budget_id or request.task_id or request.session_id),
            structured_output=request.structured_output,
            thinking=request.thinking,
            max_steps=request.max_steps,
            limits=request.limits,
            retry_policy=self._effective_retry_policy(request.retry_policy),
            tool_capability_ceiling=effective_tool_capability_ceiling,
            tool_completion=request.tool_completion,
        )
        model_failover = None
        if request.failover is None:
            execution_profile = resolve_profile(
                provider_name=registered_provider.name,
                registered_provider=registered_provider,
                model=model,
            )
        else:

            def resolve_candidate_profile(
                candidate: runtime_records.RegisteredProvider, candidate_model: str
            ) -> ExecutionProfileIdentity:
                return resolve_profile(
                    provider_name=candidate.name,
                    registered_provider=candidate,
                    model=candidate_model,
                )

            model_failover = execution_profile_admission.resolve_model_failover_execution_profile(
                policy=request.failover,
                primary=ModelTarget(provider_name=registered_provider.name, model=model),
                registered_primary=registered_provider,
                resolve_provider=self._get_registered_provider,
                resolve_candidate_profile=resolve_candidate_profile,
                redactor=self._secret_redactor,
            )
            execution_profile = model_failover.profile
        self._require_targeted_tool_delivery_capacity(
            targeted_tool_grants,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            model=model,
            model_failover=model_failover,
        )
        unavailable_profile_components = unavailable_execution_profile_components(execution_profile)
        if unavailable_profile_components:
            names = ", ".join(component.value for component in unavailable_profile_components)
            raise RuntimeError(
                "Invocation execution profile has unavailable required components: " + names
            )
        if expected_execution_profile is not None and (
            execution_profile != expected_execution_profile
        ):
            session_id = request.session_id
            if session_id is None:
                raise AssertionError("Run request session identity was not retained.")
            raise ExecutionProfileMismatchError(
                session_id=session_id,
                expected_profile_fingerprint=expected_execution_profile.fingerprint,
                candidate_profile_fingerprint=execution_profile.fingerprint,
                changed_component_classes=changed_execution_profile_components(
                    expected_execution_profile,
                    execution_profile,
                ),
                expected_profile=expected_execution_profile,
                candidate_profile=execution_profile,
            )
        # Native schema validation is provider-owned code, but it is not an
        # execution attempt. Freeze the invocation profile first while retaining
        # the established no-session-on-invalid-schema entry-point contract.
        candidate_configurations = (
            ((registered_provider, model),)
            if model_failover is None
            else tuple(
                zip(
                    model_failover.registered_providers,
                    (target.model for target in model_failover.plan.candidates),
                    strict=True,
                )
            )
        )
        for candidate_provider, candidate_model in candidate_configurations:
            if candidate_provider is not registered_provider or candidate_model != model:
                candidate_provider.provider.preflight_model_target(model=candidate_model)
            _require_native_structured_output_support(
                request.structured_output, registered_provider=candidate_provider
            )
        prepared_session_id = request.session_id
        if prepared_session_id is None:
            raise AssertionError("Run request session identity was not assigned.")
        if not inspect_only and runtime_prepared_session_authority(request) is None:
            existing_session = await self.session_store.load(prepared_session_id)
            if (
                existing_session is not None
                and not allow_existing_session_id
                and (store_resolved_existing_session_id != prepared_session_id)
            ):
                del existing_session
                raise ValueError(
                    f"Session already exists: {prepared_session_id}. "
                    "app.run creates a new session. For ordinary conversation continuation, "
                    "use app.resume(ResumeRequest(session_id=existing_id, "
                    "messages=[Message.text('user', 'Next turn')])). "
                    "Resume is subject to admission: running sessions, pending approval/input, "
                    "or incomplete work may require their resolution or recovery API first. "
                    "See `cayu guide references#sessions` and `cayu guide durable-service-tools`."
                )
        for candidate_provider, candidate_model in candidate_configurations:
            candidate_provider.provider.preflight_hosted_tools(
                model=candidate_model,
                hosted_tools=registered_agent.hosted_tools,
                options=copy_json_value(
                    registered_agent.spec.provider_options,
                    "agent provider_options",
                ),
            )
            if model_failover is not None:
                preflight_model_thinking(
                    provider=candidate_provider.provider,
                    model=candidate_model,
                    thinking=(
                        request.thinking
                        if request.thinking is not None
                        else registered_agent.spec.thinking
                        if registered_agent.spec.thinking is not None
                        else registered_agent.spec.provider_options.get("thinking")
                    ),
                    redactor=self._secret_redactor,
                )
                preflight_portable_model_material(
                    provider=candidate_provider.provider,
                    model=candidate_model,
                    messages=_model_request_messages(
                        messages=[
                            *(
                                ()
                                if rendered_system_prompt is None
                                else (Message.text("system", rendered_system_prompt),)
                            ),
                            *request.messages,
                        ],
                        structured_output=request.structured_output,
                    ),
                    tools=_model_request_tools(
                        tool_exposure=_tool_capability_ceiling_exposure(
                            registered_agent, effective_tool_capability_ceiling.tool_names
                        ),
                        structured_output=request.structured_output,
                        targeted_tool_projection=resolve_targeted_tool_projection(
                            registered_agent.targeted_tool_mode,
                            provider=candidate_provider.provider,
                            model=candidate_model,
                        ),
                        tool_discovery_mode=registered_agent.tool_discovery_mode,
                    ),
                    redactor=self._secret_redactor,
                )
        session_identity = _session_identity(
            provider_name=registered_provider.name,
            model=model,
            execution_profile=execution_profile,
        )
        if not allow_work_attempt_admission:
            (
                requires_completion_decision,
                admission_failure,
            ) = await verifier_aware_task_execution_outcome(
                task_store=self.task_store,
                redactor=self._secret_redactor,
                task_id=request.task_id,
                session_id=contract_lookup_session_id,
                admit_session=admit_session,
                allow_missing_task=deferred_runtime_task_creation,
            )
            if admission_failure is not None:
                del request, workspace_instructions, rendered_system_prompt, prompt_contributions
                raise_task_store_operation_failure(admission_failure)
            if requires_completion_decision:
                return None
        return _PreparedInitialRun(
            request=request,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            registered_environment=registered_environment,
            workspace_instructions=workspace_instructions,
            rendered_system_prompt=rendered_system_prompt,
            prompt_contributions=tuple(prompt_contributions),
            execution_profile=execution_profile,
            model_failover=model_failover,
            tool_capability_ceiling=effective_tool_capability_ceiling,
            targeted_tool_grants=targeted_tool_grants,
            budget_policy=budget_policy,
            session_identity=session_identity,
            policy_evidence=policy_evidence,
            managed_settings_evidence=(
                None
                if self._managed_evidence is None
                else self._managed_evidence(
                    registered_agent.spec.name,
                    {
                        # What this session actually runs with; only Cloud values that
                        # match it are recorded (explicit fields, retained models and
                        # queued children may differ from Cloud's latest values).
                        # None marks a value the request set itself, which isn't Cloud's.
                        "config.run.max_steps": (
                            None if "max_steps" in explicit_fields else request.max_steps
                        ),
                        **{
                            f"config.run.limits.{name}": (
                                None if "limits" in explicit_fields else value
                            )
                            for name, value in request.limits.model_dump().items()
                        },
                        "config.run.retry_policy.max_attempts": (
                            None
                            if "retry_policy" in explicit_fields
                            else self._effective_retry_policy(request.retry_policy).max_attempts
                        ),
                        f"agent.{registered_agent.spec.name}.model": (
                            None
                            if policy_evidence is not None or request.target is not None
                            else model
                        ),
                    },
                )
            ),
        )
