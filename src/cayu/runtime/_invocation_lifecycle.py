"""Frozen invocation authority and the versioned session-store command seam.

This module is deliberately deeper than the runtime facades which consume it.
It owns the in-process authority bundle and the finite command family used to
mutate durable active-invocation state.  Live Python collaborators are never
serialized by this module; only the command values cross a store boundary.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError, dataclass
from enum import StrEnum
from hashlib import sha256
from typing import Annotated, Any, Literal, Never, SupportsIndex, TypeAlias, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    StrictBool,
    StrictInt,
    TypeAdapter,
    field_validator,
    model_validator,
)

from cayu._validation import (
    DurableValueError,
    canonical_bounded_durable_json_bytes,
    canonical_durable_json_bytes,
    copy_durable_json_object,
    require_durable_clean_nonblank,
)
from cayu.core.events import (
    Event,
    EventType,
    copy_event,
    event_envelope_authority_is_runtime_generated,
)
from cayu.core.messages import Message, detach_message
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime.budgets import BudgetPolicy
from cayu.runtime.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
    INVOCATION_LIFECYCLE_RECEIPT_LEDGER_RECORD_TYPE,
    INVOCATION_LIFECYCLE_RECEIPT_LEDGER_SCHEMA_VERSION,
)
from cayu.runtime.execution_profiles import (
    ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
    ActiveInvocationExecutionProfile,
    ExecutionProfileDecision,
    ExecutionProfileDecisionKind,
    ExecutionProfileIdentity,
    active_invocation_execution_profile_from_checkpoint,
    changed_execution_profile_components,
    checkpoint_with_active_invocation_execution_profile,
    direct_tool_capability_ceiling_component,
    execution_profile_baseline_from_session_metadata,
    execution_profile_changes_authority,
    execution_profile_from_session_metadata,
    execution_profile_provider_target_component,
    execution_profile_runtime_component,
)
from cayu.runtime.invocation import SessionInvocationBinding
from cayu.runtime.loop_policies import LoopPolicy
from cayu.runtime.sessions import (
    ExecutionProfileRejectionResult,
    InteractionTransitionResult,
    InteractionTransitionSpec,
    RunRequest,
    RuntimePublicationMutation,
    Session,
    SessionIdentity,
    SessionInvocationAdmission,
    SessionModelTransition,
    SessionRunFenced,
    SessionStatus,
    SessionStore,
    _authenticated_session_instance_id_for_run_request,
    _invocation_lifecycle_authority_mutation_scope,
    _run_request_invocation_lifecycle_authority_sha256,
    apply_runtime_publication_checkpoint_mutation,
    copy_run_request,
    copy_session,
    copy_session_identity,
    runtime_prepared_session_authority,
    runtime_publication_checkpoint_mutation,
    session_user_metadata,
)
from cayu.runtime.tool_discovery import (
    ToolDiscoveryViewInitialization,
    initial_tool_discovery_operation_records_from_initialization,
)
from cayu.runtime.tool_exposure import (
    ToolCapabilityCeiling,
    copy_tool_capability_ceiling,
    tool_capability_ceiling_from_session_metadata,
)
from cayu.runtime.tool_grants import PreparedTargetedToolGrant

INVOCATION_LIFECYCLE_COMMAND_VERSION = 1
INVOCATION_LIFECYCLE_RECEIPT_LEDGER_MAX_ITEMS = 128
INVOCATION_LIFECYCLE_RECEIPT_LEDGER_MAX_BYTES = 8 * 1024 * 1024
_INVOCATION_LIFECYCLE_RECEIPT_LEDGER_MAX_NODES = 131_072
_INVOCATION_LIFECYCLE_RECEIPT_RECORD_TYPE = "cayu.invocation-lifecycle-command-receipt"
_INVOCATION_LIFECYCLE_RECEIPT_SCHEMA_VERSION = 1
_RELEASE_CLEANUP_AUTHORITY_TOKEN = object()
_RELEASE_STORE_AUTHORITY_TOKEN = object()
_INVOCATION_CONTEXT_AUTHORITY_TOKEN = object()


class InvocationLifecycleCommandKind(StrEnum):
    """Closed command family understood by lifecycle command version 1."""

    CREATE = "create"
    ADMIT = "admit"
    REBIND = "rebind"
    REJECT = "reject"
    SETTLE = "settle"
    RELEASE = "release"


class InvocationLifecycleCommandConflict(ValueError):
    """A stable lifecycle command identity names different authority."""


def _validate_invocation_binding(binding: Any) -> None:
    for field_name in (
        "session_id",
        "interaction_id",
        "agent_name",
        "provider_name",
        "model",
        "runtime_name",
    ):
        object.__setattr__(
            binding,
            field_name,
            require_durable_clean_nonblank(getattr(binding, field_name), field_name),
        )
    object.__setattr__(
        binding,
        "session_instance_id",
        SessionInvocationBinding.validate_session_instance_id(binding.session_instance_id),
    )
    run_epoch = binding.run_epoch
    if type(run_epoch) is not int or run_epoch < 1:
        raise ValueError("run_epoch must be a positive integer.")
    for field_name in ("runtime_version", "environment_name"):
        value = getattr(binding, field_name)
        if value is not None:
            object.__setattr__(
                binding,
                field_name,
                require_durable_clean_nonblank(value, field_name),
            )


@dataclass(frozen=True, slots=True)
class PreparedInvocationBinding:
    """A store target fixed before one invocation is durably admitted."""

    session_id: str
    session_instance_id: str
    interaction_id: str
    run_epoch: int
    agent_name: str
    provider_name: str
    model: str
    runtime_name: str
    runtime_version: str | None
    environment_name: str | None

    def __post_init__(self) -> None:
        _validate_invocation_binding(self)


@dataclass(frozen=True, slots=True)
class AdmittedInvocationBinding:
    """Immutable durable authority projected from an admitted session."""

    session_id: str
    session_instance_id: str
    interaction_id: str
    run_epoch: int
    agent_name: str
    provider_name: str
    model: str
    runtime_name: str
    runtime_version: str | None
    environment_name: str | None

    def __post_init__(self) -> None:
        _validate_invocation_binding(self)


InvocationBinding: TypeAlias = PreparedInvocationBinding | AdmittedInvocationBinding


class InvocationContext:
    """One immutable live authority bundle for an invocation.

    Registered collaborators are retained by identity.  The context is an
    in-process value and intentionally has no serialization API.
    """

    __slots__ = (
        "_active_profile",
        "_authority_token",
        "_binding",
        "_budget_policy",
        "_loop_policies",
        "_registered_agent",
        "_registered_environment",
        "_registered_provider",
        "_request_loop_policies",
        "_runtime_hooks",
        "_targeted_tool_grants",
        "_tool_capability_ceiling",
        "_validated_profile",
    )

    _active_profile: ActiveInvocationExecutionProfile
    _authority_token: object
    _binding: InvocationBinding
    _budget_policy: BudgetPolicy | None
    _loop_policies: tuple[LoopPolicy, ...]
    _registered_agent: runtime_records.RegisteredAgentState
    _registered_environment: runtime_records.RegisteredEnvironment | None
    _registered_provider: runtime_records.RegisteredProvider
    _request_loop_policies: tuple[LoopPolicy, ...]
    _runtime_hooks: tuple[runtime_records.RegisteredRuntimeHook, ...]
    _targeted_tool_grants: tuple[PreparedTargetedToolGrant, ...]
    _tool_capability_ceiling: ToolCapabilityCeiling
    _validated_profile: ExecutionProfileIdentity

    def __init__(
        self,
        *,
        active_profile: ActiveInvocationExecutionProfile,
        binding: InvocationBinding,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        runtime_hooks: tuple[runtime_records.RegisteredRuntimeHook, ...],
        loop_policies: tuple[LoopPolicy, ...],
        request_loop_policies: tuple[LoopPolicy, ...],
        budget_policy: BudgetPolicy | None,
        tool_capability_ceiling: ToolCapabilityCeiling,
        targeted_tool_grants: tuple[PreparedTargetedToolGrant, ...] = (),
        _validated_profile: ExecutionProfileIdentity | None = None,
        _authority_token: object = None,
    ) -> None:
        object.__setattr__(self, "_active_profile", active_profile)
        object.__setattr__(self, "_binding", binding)
        object.__setattr__(self, "_registered_agent", registered_agent)
        object.__setattr__(self, "_registered_provider", registered_provider)
        object.__setattr__(self, "_registered_environment", registered_environment)
        object.__setattr__(self, "_runtime_hooks", runtime_hooks)
        object.__setattr__(self, "_loop_policies", loop_policies)
        object.__setattr__(self, "_request_loop_policies", request_loop_policies)
        object.__setattr__(self, "_budget_policy", budget_policy)
        object.__setattr__(self, "_tool_capability_ceiling", tool_capability_ceiling)
        object.__setattr__(self, "_targeted_tool_grants", targeted_tool_grants)
        object.__setattr__(self, "_validated_profile", _validated_profile)
        object.__setattr__(self, "_authority_token", _authority_token)
        self._validate()

    def _validate(self) -> None:
        if self._authority_token is not _INVOCATION_CONTEXT_AUTHORITY_TOKEN:
            raise TypeError(
                "InvocationContext must be constructed by the runtime authority boundary."
            )
        if type(self.active_profile) is not ActiveInvocationExecutionProfile:
            raise TypeError("active_profile must be an ActiveInvocationExecutionProfile.")
        if type(self.binding) not in (PreparedInvocationBinding, AdmittedInvocationBinding):
            raise TypeError("binding must be a prepared or admitted invocation binding.")
        if type(self.registered_agent) is not runtime_records.RegisteredAgentState:
            raise TypeError("registered_agent must be a RegisteredAgentState.")
        if type(self.registered_provider) is not runtime_records.RegisteredProvider:
            raise TypeError("registered_provider must be a RegisteredProvider.")
        if self.registered_environment is not None and (
            type(self.registered_environment) is not runtime_records.RegisteredEnvironment
        ):
            raise TypeError("registered_environment must be a RegisteredEnvironment or None.")
        if type(self.runtime_hooks) is not tuple or any(
            type(hook) is not runtime_records.RegisteredRuntimeHook for hook in self.runtime_hooks
        ):
            raise TypeError("runtime_hooks must contain RegisteredRuntimeHook values.")
        if type(self.loop_policies) is not tuple or any(
            not isinstance(policy, LoopPolicy) for policy in self.loop_policies
        ):
            raise TypeError("loop_policies must contain LoopPolicy values.")
        if type(self.request_loop_policies) is not tuple or any(
            not isinstance(policy, LoopPolicy) for policy in self.request_loop_policies
        ):
            raise TypeError("request_loop_policies must contain LoopPolicy values.")
        if self.budget_policy is not None and type(self.budget_policy) is not BudgetPolicy:
            raise TypeError("budget_policy must be a BudgetPolicy or None.")
        if type(self.tool_capability_ceiling) is not ToolCapabilityCeiling:
            raise TypeError("tool_capability_ceiling must be a ToolCapabilityCeiling.")
        if type(self.targeted_tool_grants) is not tuple or any(
            type(grant) is not PreparedTargetedToolGrant for grant in self.targeted_tool_grants
        ):
            raise TypeError("targeted_tool_grants must contain PreparedTargetedToolGrant values.")
        if type(self._validated_profile) is not ExecutionProfileIdentity:
            raise TypeError("Invocation context lacks a validated execution profile.")
        if self._validated_profile is not self.active_profile.profile:
            raise ValueError(
                "Live invocation authority must retain the exact validated profile object."
            )

        binding = self.binding
        if self.active_profile.session_id != binding.session_id:
            raise ValueError("Invocation context profile belongs to another session.")
        if self.active_profile.interaction_id != binding.interaction_id:
            raise ValueError("Invocation context profile belongs to another interaction.")
        if self.active_profile.run_epoch != binding.run_epoch:
            raise ValueError("Invocation context profile belongs to another run epoch.")
        if self.registered_agent.spec.name != binding.agent_name:
            raise ValueError("Registered agent conflicts with invocation authority.")
        if self.registered_provider.name != binding.provider_name:
            raise ValueError("Registered provider conflicts with invocation authority.")
        environment_name = (
            None if self.registered_environment is None else self.registered_environment.spec.name
        )
        if environment_name is not None and environment_name != binding.environment_name:
            raise ValueError("Registered environment conflicts with invocation authority.")
        if binding.environment_name is None and environment_name is not None:
            raise ValueError("Invocation authority does not permit an environment.")
        provider_component = self.profile.component(
            execution_profile_provider_target_component(
                binding.provider_name,
                binding.model,
            ).component_class
        )
        if provider_component != execution_profile_provider_target_component(
            binding.provider_name,
            binding.model,
        ):
            raise ValueError("Invocation profile conflicts with provider/model authority.")
        ceiling_component = direct_tool_capability_ceiling_component(
            self.tool_capability_ceiling.tool_names
        )
        if self.profile.component(ceiling_component.component_class) != ceiling_component:
            raise ValueError("Invocation profile conflicts with tool ceiling authority.")

    def __setattr__(self, _name: str, _value: object) -> None:
        raise FrozenInstanceError("InvocationContext is immutable.")

    def __copy__(self) -> InvocationContext:
        return self

    def __deepcopy__(self, _memo: dict[int, object]) -> InvocationContext:
        return self

    def __reduce_ex__(self, _protocol: SupportsIndex, /) -> Never:
        raise TypeError("InvocationContext has no serialization form.")

    def __repr__(self) -> str:
        """Keep live collaborators out of traceback-local diagnostics."""

        return "InvocationContext(<authenticated>)"

    @property
    def profile(self) -> ExecutionProfileIdentity:
        """Return the sole validated profile object owned by this context."""

        return self._validated_profile

    @property
    def active_profile(self) -> ActiveInvocationExecutionProfile:
        return self._active_profile

    @property
    def binding(self) -> InvocationBinding:
        return self._binding

    @property
    def registered_agent(self) -> runtime_records.RegisteredAgentState:
        return self._registered_agent

    @property
    def registered_provider(self) -> runtime_records.RegisteredProvider:
        return self._registered_provider

    @property
    def registered_environment(self) -> runtime_records.RegisteredEnvironment | None:
        return self._registered_environment

    @property
    def runtime_hooks(self) -> tuple[runtime_records.RegisteredRuntimeHook, ...]:
        return self._runtime_hooks

    @property
    def loop_policies(self) -> tuple[LoopPolicy, ...]:
        return self._loop_policies

    @property
    def request_loop_policies(self) -> tuple[LoopPolicy, ...]:
        return self._request_loop_policies

    @property
    def budget_policy(self) -> BudgetPolicy | None:
        return self._budget_policy

    @property
    def tool_capability_ceiling(self) -> ToolCapabilityCeiling:
        return self._tool_capability_ceiling

    @property
    def targeted_tool_grants(self) -> tuple[PreparedTargetedToolGrant, ...]:
        return self._targeted_tool_grants

    def with_admitted_session(self, session: Session) -> InvocationContext:
        """Attach the durable admission result without replacing live authority."""

        if type(session) is not Session:
            raise TypeError("session must be a Session.")
        binding = self.binding
        if isinstance(binding, AdmittedInvocationBinding):
            if (
                session.id != binding.session_id
                or session.instance_id != binding.session_instance_id
                or session.run_epoch != binding.run_epoch
                or session.agent_name != binding.agent_name
                or session.provider_name != binding.provider_name
                or session.model != binding.model
                or session.runtime_name != binding.runtime_name
                or session.runtime_version != binding.runtime_version
                or session.environment_name != binding.environment_name
            ):
                raise ValueError("Invocation context cannot replace its admitted session.")
            return self
        if (
            session.id != binding.session_id
            or session.instance_id != binding.session_instance_id
            or session.run_epoch != binding.run_epoch
            or session.agent_name != binding.agent_name
            or session.provider_name != binding.provider_name
            or session.model != binding.model
            or session.runtime_name != binding.runtime_name
            or session.runtime_version != binding.runtime_version
            or session.environment_name != binding.environment_name
        ):
            raise ValueError("Admitted session conflicts with prepared invocation authority.")
        return _authenticated_invocation_context(
            active_profile=self.active_profile,
            binding=AdmittedInvocationBinding(
                session_id=binding.session_id,
                session_instance_id=binding.session_instance_id,
                interaction_id=binding.interaction_id,
                run_epoch=binding.run_epoch,
                agent_name=binding.agent_name,
                provider_name=binding.provider_name,
                model=binding.model,
                runtime_name=binding.runtime_name,
                runtime_version=binding.runtime_version,
                environment_name=binding.environment_name,
            ),
            validated_profile=self.profile,
            registered_agent=self.registered_agent,
            registered_provider=self.registered_provider,
            registered_environment=self.registered_environment,
            runtime_hooks=self.runtime_hooks,
            loop_policies=self.loop_policies,
            request_loop_policies=self.request_loop_policies,
            budget_policy=self.budget_policy,
            tool_capability_ceiling=self.tool_capability_ceiling,
            targeted_tool_grants=self.targeted_tool_grants,
        )

    def with_registered_environment(
        self,
        registered_environment: runtime_records.RegisteredEnvironment,
        *,
        validated_profile: ExecutionProfileIdentity,
    ) -> InvocationContext:
        """Transfer one post-admission materialized environment into the context."""

        if type(registered_environment) is not runtime_records.RegisteredEnvironment:
            raise TypeError("registered_environment must be a RegisteredEnvironment.")
        if validated_profile is not self.profile:
            raise ValueError("Environment transfer must retain the exact validated profile object.")
        if self.registered_environment is not None:
            current = self.registered_environment
            if current is registered_environment:
                return self
            if (
                current.spec != registered_environment.spec
                or current.runner_execution_profile_identity
                != registered_environment.runner_execution_profile_identity
                or current.factory_execution_profile_identity
                != registered_environment.factory_execution_profile_identity
                or current.factory_backed != registered_environment.factory_backed
                or current.registration_source != registered_environment.registration_source
                or current.registration_symbol != registered_environment.registration_symbol
            ):
                raise ValueError("Invocation context cannot replace its resolved environment.")
            if current.factory is not None:
                if (
                    not current.factory_backed
                    or registered_environment.factory is not None
                    or registered_environment.unclaimed_factory_result is None
                    or registered_environment.bound_workspace is not None
                    or registered_environment.retained_factory_result is not None
                    or registered_environment.preserve_factory_allocation
                    or registered_environment.binding_generation_id == current.binding_generation_id
                    or registered_environment.workspace_mutation_fence
                    is current.workspace_mutation_fence
                ):
                    raise ValueError(
                        "Invocation context received an invalid factory materialization."
                    )
            else:
                self._require_monotonic_environment_resolution(
                    current,
                    registered_environment,
                )
            return _authenticated_invocation_context(
                active_profile=self.active_profile,
                binding=self.binding,
                validated_profile=self.profile,
                registered_agent=self.registered_agent,
                registered_provider=self.registered_provider,
                registered_environment=registered_environment,
                runtime_hooks=self.runtime_hooks,
                loop_policies=self.loop_policies,
                request_loop_policies=self.request_loop_policies,
                budget_policy=self.budget_policy,
                tool_capability_ceiling=self.tool_capability_ceiling,
                targeted_tool_grants=self.targeted_tool_grants,
            )
        if self.binding.environment_name is None:
            raise ValueError("Invocation authority does not permit an environment.")
        if registered_environment.spec.name != self.binding.environment_name:
            raise ValueError("Registered environment conflicts with invocation authority.")
        return _authenticated_invocation_context(
            active_profile=self.active_profile,
            binding=self.binding,
            validated_profile=self.profile,
            registered_agent=self.registered_agent,
            registered_provider=self.registered_provider,
            registered_environment=registered_environment,
            runtime_hooks=self.runtime_hooks,
            loop_policies=self.loop_policies,
            request_loop_policies=self.request_loop_policies,
            budget_policy=self.budget_policy,
            tool_capability_ceiling=self.tool_capability_ceiling,
            targeted_tool_grants=self.targeted_tool_grants,
        )

    @staticmethod
    def _require_monotonic_environment_resolution(
        current: runtime_records.RegisteredEnvironment,
        replacement: runtime_records.RegisteredEnvironment,
    ) -> None:
        """Accept only binding or owned-result settlement, never handle substitution."""

        if (
            replacement.factory is not None
            or replacement.binding_generation_id != current.binding_generation_id
            or replacement.workspace_mutation_fence is not current.workspace_mutation_fence
            or replacement.execution_candidate != current.execution_candidate
            or replacement.live_allocation_fingerprint != current.live_allocation_fingerprint
        ):
            raise ValueError("Invocation context cannot replace its resolved environment.")

        if current.bound_workspace is not None:
            current_environment = current.environment
            replacement_environment = replacement.environment
            if (
                replacement.bound_workspace is not current.bound_workspace
                or replacement_environment.workspace is not current_environment.workspace
                or replacement_environment.runner is not current_environment.runner
                or replacement_environment.artifact_store is not current_environment.artifact_store
                or replacement_environment.vault is not current_environment.vault
                or replacement_environment.proxy is not current_environment.proxy
                or replacement_environment.knowledge_store
                is not current_environment.knowledge_store
                or replacement_environment.binding is not current_environment.binding
                or replacement_environment.mcp_servers != current_environment.mcp_servers
                or replacement_environment.workspace_instructions
                != current_environment.workspace_instructions
                or replacement.binding_payload is not current.binding_payload
                or replacement.retained_factory_result is not current.retained_factory_result
                or (
                    current.unclaimed_factory_result is None
                    and replacement.unclaimed_factory_result is not None
                )
                or (
                    current.unclaimed_factory_result is not None
                    and replacement.unclaimed_factory_result is not None
                    and replacement.unclaimed_factory_result is not current.unclaimed_factory_result
                )
                or (
                    not current.preserve_factory_allocation
                    and replacement.preserve_factory_allocation
                )
            ):
                raise ValueError("Invocation context cannot replace its bound environment.")
            return

        if replacement.bound_workspace is None:
            if (
                replacement.environment is not current.environment
                or current.bound_workspace is not None
                or replacement.binding_payload is not current.binding_payload
                or replacement.retained_factory_result is not current.retained_factory_result
                or replacement.preserve_factory_allocation != current.preserve_factory_allocation
                or (
                    current.unclaimed_factory_result is None
                    and replacement.unclaimed_factory_result is not None
                )
                or (
                    current.unclaimed_factory_result is not None
                    and replacement.unclaimed_factory_result is not None
                    and replacement.unclaimed_factory_result is not current.unclaimed_factory_result
                )
            ):
                raise ValueError(
                    "Invocation context received an invalid environment-owner settlement."
                )
            return

        bound = replacement.bound_workspace
        environment = replacement.environment
        current_environment = current.environment
        if (
            current.bound_workspace is not None
            or current_environment.binding is None
            or current.binding_payload is not None
            or type(replacement.binding_payload) is not dict
            or replacement.unclaimed_factory_result is not None
            or replacement.retained_factory_result is not current.unclaimed_factory_result
            or type(replacement.preserve_factory_allocation) is not bool
            or (
                current.unclaimed_factory_result is None and replacement.preserve_factory_allocation
            )
            or environment.workspace is not bound.workspace
            or environment.runner is not bound.runner
            or environment.artifact_store is not current_environment.artifact_store
            or environment.vault is not current_environment.vault
            or environment.proxy is not current_environment.proxy
            or environment.knowledge_store is not current_environment.knowledge_store
            or environment.binding is not current_environment.binding
            or environment.mcp_servers != current_environment.mcp_servers
            or environment.workspace_instructions != current_environment.workspace_instructions
        ):
            raise ValueError("Invocation context received an invalid workspace binding.")

    def with_rebound_session(
        self,
        session: Session,
        *,
        active_profile: ActiveInvocationExecutionProfile,
    ) -> InvocationContext:
        """Carry the same live authority through one authenticated epoch rebind."""

        if type(session) is not Session:
            raise TypeError("session must be a Session.")
        if type(active_profile) is not ActiveInvocationExecutionProfile:
            raise TypeError("active_profile must be an ActiveInvocationExecutionProfile.")
        binding = self.binding
        if (
            session.id != binding.session_id
            or session.instance_id != binding.session_instance_id
            or session.agent_name != binding.agent_name
            or session.provider_name != binding.provider_name
            or session.model != binding.model
            or session.runtime_name != binding.runtime_name
            or session.runtime_version != binding.runtime_version
            or session.environment_name != binding.environment_name
            or active_profile.session_id != session.id
            or active_profile.interaction_id != binding.interaction_id
            or active_profile.run_epoch != session.run_epoch
            or active_profile.profile is not self.profile
        ):
            raise ValueError("Rebound session conflicts with invocation authority.")
        return _authenticated_invocation_context(
            active_profile=active_profile,
            binding=AdmittedInvocationBinding(
                session_id=session.id,
                session_instance_id=session.instance_id,
                interaction_id=active_profile.interaction_id,
                run_epoch=session.run_epoch,
                agent_name=session.agent_name,
                provider_name=session.provider_name,
                model=session.model,
                runtime_name=session.runtime_name,
                runtime_version=session.runtime_version,
                environment_name=session.environment_name,
            ),
            validated_profile=self.profile,
            registered_agent=self.registered_agent,
            registered_provider=self.registered_provider,
            registered_environment=self.registered_environment,
            runtime_hooks=self.runtime_hooks,
            loop_policies=self.loop_policies,
            request_loop_policies=self.request_loop_policies,
            budget_policy=self.budget_policy,
            tool_capability_ceiling=self.tool_capability_ceiling,
            targeted_tool_grants=self.targeted_tool_grants,
        )


def _authenticated_invocation_context(
    *,
    active_profile: ActiveInvocationExecutionProfile,
    binding: InvocationBinding,
    validated_profile: ExecutionProfileIdentity,
    registered_agent: runtime_records.RegisteredAgentState,
    registered_provider: runtime_records.RegisteredProvider,
    registered_environment: runtime_records.RegisteredEnvironment | None,
    runtime_hooks: tuple[runtime_records.RegisteredRuntimeHook, ...],
    loop_policies: tuple[LoopPolicy, ...],
    request_loop_policies: tuple[LoopPolicy, ...],
    budget_policy: BudgetPolicy | None,
    tool_capability_ceiling: ToolCapabilityCeiling,
    targeted_tool_grants: tuple[PreparedTargetedToolGrant, ...] = (),
) -> InvocationContext:
    """Authenticate independently resolved live collaborators against one profile."""

    if type(validated_profile) is not ExecutionProfileIdentity:
        raise TypeError("validated_profile must be an ExecutionProfileIdentity.")
    return InvocationContext(
        active_profile=active_profile,
        binding=binding,
        registered_agent=registered_agent,
        registered_provider=registered_provider,
        registered_environment=registered_environment,
        runtime_hooks=runtime_hooks,
        loop_policies=loop_policies,
        request_loop_policies=request_loop_policies,
        budget_policy=budget_policy,
        tool_capability_ceiling=tool_capability_ceiling,
        targeted_tool_grants=targeted_tool_grants,
        _validated_profile=validated_profile,
        _authority_token=_INVOCATION_CONTEXT_AUTHORITY_TOKEN,
    )


class InvocationCheckpointPatch(BaseModel):
    """A bounded root-level CAS patch that cannot address invocation authority."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    mutation: RuntimePublicationMutation = Field(default_factory=RuntimePublicationMutation)

    @field_validator("mutation", mode="before")
    @classmethod
    def copy_mutation(cls, value: object) -> RuntimePublicationMutation:
        if isinstance(value, InvocationCheckpointPatch):
            value = value.mutation
        if isinstance(value, RuntimePublicationMutation):
            value = value.model_dump(mode="python")
        return RuntimePublicationMutation.model_validate(value)

    @model_validator(mode="after")
    def reject_authority_operations(self) -> InvocationCheckpointPatch:
        if any(
            operation.key
            in {
                CHECKPOINT_SCHEMA_VERSION_KEY,
                ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
                INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
            }
            for operation in self.mutation.operations
        ):
            raise ValueError("Invocation checkpoint patches cannot mutate lifecycle authority.")
        return self


class _InvocationCommandModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    schema_version: Literal[1] = INVOCATION_LIFECYCLE_COMMAND_VERSION
    session_id: str
    expected_session_instance_id: str

    @field_validator("session_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("expected_session_instance_id")
    @classmethod
    def validate_session_instance_id(cls, value: str) -> str:
        return SessionInvocationBinding.validate_session_instance_id(value)


class CreateInvocationCommand(_InvocationCommandModel):
    kind: Literal[InvocationLifecycleCommandKind.CREATE] = InvocationLifecycleCommandKind.CREATE
    request: RunRequest
    identity: SessionIdentity
    active_profile: ActiveInvocationExecutionProfile
    interaction_started_event: Event
    interaction_source_messages: tuple[Message, ...]
    checkpoint_patch: InvocationCheckpointPatch = Field(default_factory=InvocationCheckpointPatch)
    tool_discovery_initialization: ToolDiscoveryViewInitialization | None = None

    @field_validator("request", mode="before")
    @classmethod
    def copy_request(cls, value: RunRequest) -> RunRequest:
        return copy_run_request(value)

    @field_validator("identity", mode="before")
    @classmethod
    def copy_identity(cls, value: SessionIdentity) -> SessionIdentity:
        return copy_session_identity(value)

    @field_validator("interaction_started_event", mode="before")
    @classmethod
    def copy_event(cls, value: Event) -> Event:
        return copy_event(value)

    @field_validator("interaction_source_messages", mode="before")
    @classmethod
    def copy_messages(cls, value: object) -> tuple[Message, ...]:
        if type(value) not in (tuple, list):
            raise TypeError("interaction_source_messages must be a tuple or list.")
        items = cast("tuple[object, ...] | list[object]", value)
        if any(not isinstance(item, Message) for item in items):
            raise TypeError("interaction_source_messages must contain Message values.")
        return tuple(detach_message(cast("Message", item)) for item in items)

    @model_validator(mode="after")
    def validate_create_authority(self) -> CreateInvocationCommand:
        if self.request.session_id != self.session_id:
            raise ValueError("Create invocation request must carry the exact session ID.")
        if runtime_prepared_session_authority(self.request) is not None:
            raise ValueError("Create invocation cannot publish a prepared pending session.")
        if self.request.target is not None and (
            self.request.target.provider_name != self.identity.provider_name
            or self.request.target.model != self.identity.model
        ):
            raise ValueError("Create invocation request target conflicts with session identity.")
        if (
            _authenticated_session_instance_id_for_run_request(
                self.request,
                session_id=self.session_id,
            )
            != self.expected_session_instance_id
        ):
            raise ValueError(
                "Create invocation request lacks the exact authenticated session incarnation."
            )
        event = self.interaction_started_event
        if event.session_id != self.session_id or event.interaction_id is None:
            raise ValueError("Create invocation event belongs to another session or interaction.")
        if event.type is not EventType.INTERACTION_STARTED:
            raise ValueError("Create invocation requires an interaction-started event.")
        if (
            event.agent_name != self.request.agent_name
            or event.environment_name != self.request.environment_name
        ):
            raise ValueError("Create invocation event conflicts with request authority.")
        if tuple(self.request.messages) != self.interaction_source_messages:
            raise ValueError("Create invocation source messages conflict with the request.")
        if self.active_profile.session_id != self.session_id:
            raise ValueError("Create invocation profile belongs to another session.")
        if self.active_profile.interaction_id != event.interaction_id:
            raise ValueError("Create invocation profile belongs to another interaction.")
        if self.active_profile.run_epoch != 1:
            raise ValueError("A newly created invocation must own run epoch 1.")
        if self.identity.execution_profile != self.active_profile.profile:
            raise ValueError("Create invocation identity conflicts with active authority.")
        if self.request.tool_capability_ceiling is None:
            raise ValueError("Create invocation requires a resolved tool capability ceiling.")
        discovery = self.tool_discovery_initialization
        if discovery is not None and (
            discovery.session_id != self.session_id
            or discovery.agent_name != self.request.agent_name
            or discovery.ceiling_fingerprint
            != f"sha256:{self.request.tool_capability_ceiling.fingerprint}"
        ):
            raise ValueError(
                "Create invocation discovery initialization conflicts with request authority."
            )
        ceiling_component = direct_tool_capability_ceiling_component(
            self.request.tool_capability_ceiling.tool_names
        )
        if (
            self.active_profile.profile.component(ceiling_component.component_class)
            != ceiling_component
        ):
            raise ValueError("Create invocation profile conflicts with its tool ceiling.")
        provider_component = execution_profile_provider_target_component(
            self.identity.provider_name,
            self.identity.model,
        )
        if (
            self.active_profile.profile.component(provider_component.component_class)
            != provider_component
        ):
            raise ValueError("Create invocation profile conflicts with provider/model identity.")
        runtime_component = execution_profile_runtime_component(
            self.identity.runtime_name,
            self.identity.runtime_version,
        )
        if (
            self.active_profile.profile.component(runtime_component.component_class)
            != runtime_component
        ):
            raise ValueError("Create invocation profile conflicts with runtime identity.")
        return self


class AdmitInvocationCommand(_InvocationCommandModel):
    kind: Literal[InvocationLifecycleCommandKind.ADMIT] = InvocationLifecycleCommandKind.ADMIT
    expected_statuses: tuple[SessionStatus, ...]
    expected_run_epoch: StrictInt = Field(ge=0)
    expected_checkpoint_sha256: str
    target_active_profile: ActiveInvocationExecutionProfile
    checkpoint_patch: InvocationCheckpointPatch = Field(default_factory=InvocationCheckpointPatch)
    interaction_source_messages: tuple[Message, ...] = ()
    tool_capability_ceiling: ToolCapabilityCeiling
    interaction_started_event: Event | None = None
    continued_interaction_id: str | None = None
    defer_interaction_source: StrictBool = False
    model_transition: SessionModelTransition | None = None
    execution_profile_decision: ExecutionProfileDecision | None = None
    expected_active_profile: ActiveInvocationExecutionProfile | None = None
    allow_pending_initial_interaction: StrictBool = False

    @field_validator("expected_statuses", mode="before")
    @classmethod
    def copy_statuses(cls, value: object) -> tuple[SessionStatus, ...]:
        if type(value) not in (tuple, list, set, frozenset):
            raise TypeError("expected_statuses must be a status collection.")
        items = cast("tuple[object, ...] | list[object] | set[object] | frozenset[object]", value)
        statuses = tuple(sorted((SessionStatus(item) for item in items), key=str))
        if not statuses or len(set(statuses)) != len(statuses):
            raise ValueError("expected_statuses must be non-empty and unique.")
        return statuses

    @field_validator("interaction_source_messages", mode="before")
    @classmethod
    def copy_messages(cls, value: object) -> tuple[Message, ...]:
        if type(value) not in (tuple, list):
            raise TypeError("interaction_source_messages must be a tuple or list.")
        items = cast("tuple[object, ...] | list[object]", value)
        if any(not isinstance(item, Message) for item in items):
            raise TypeError("interaction_source_messages must contain Message values.")
        return tuple(detach_message(cast("Message", item)) for item in items)

    @field_validator("expected_checkpoint_sha256")
    @classmethod
    def validate_checkpoint_digest(cls, value: str) -> str:
        value = require_durable_clean_nonblank(value, "expected_checkpoint_sha256")
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("expected_checkpoint_sha256 must be lowercase SHA-256.")
        return value

    @field_validator("tool_capability_ceiling", mode="before")
    @classmethod
    def copy_ceiling(cls, value: ToolCapabilityCeiling) -> ToolCapabilityCeiling:
        return copy_tool_capability_ceiling(value)

    @field_validator("interaction_started_event", mode="before")
    @classmethod
    def copy_optional_event(cls, value: Event | None) -> Event | None:
        return None if value is None else copy_event(value)

    @field_validator("continued_interaction_id")
    @classmethod
    def validate_continued_interaction(cls, value: str | None) -> str | None:
        return (
            None
            if value is None
            else require_durable_clean_nonblank(value, "continued_interaction_id")
        )

    @model_validator(mode="after")
    def validate_admission_authority(self) -> AdmitInvocationCommand:
        target = self.target_active_profile
        if target.session_id != self.session_id:
            raise ValueError("Admission target belongs to another session.")
        if target.run_epoch != self.expected_run_epoch + 1:
            raise ValueError("Admission target must claim the next run epoch.")
        expected = self.expected_active_profile
        if expected is not None:
            if expected.session_id != self.session_id:
                raise ValueError("Admission source belongs to another session.")
            if expected.run_epoch != self.expected_run_epoch - 1:
                raise ValueError(
                    "Admission source must carry the immediately preceding released epoch."
                )
        ceiling_component = direct_tool_capability_ceiling_component(
            self.tool_capability_ceiling.tool_names
        )
        if target.profile.component(ceiling_component.component_class) != ceiling_component:
            raise ValueError("Admission target profile conflicts with its tool ceiling.")
        starts = self.interaction_started_event is not None
        continues = self.continued_interaction_id is not None
        if starts == continues:
            raise ValueError("Admission requires exactly one new or continued interaction.")
        interaction_id = (
            self.interaction_started_event.interaction_id
            if self.interaction_started_event is not None
            else self.continued_interaction_id
        )
        if interaction_id is None or target.interaction_id != interaction_id:
            raise ValueError("Admission target belongs to another interaction.")
        if self.interaction_started_event is not None and (
            self.interaction_started_event.session_id != self.session_id
        ):
            raise ValueError("Admission event belongs to another session.")
        if (
            self.interaction_started_event is not None
            and self.interaction_started_event.type is not EventType.INTERACTION_STARTED
        ):
            raise ValueError("Admission requires an interaction-started event.")
        if continues:
            if expected is None:
                raise ValueError("Continued admission requires prior active authority.")
            if expected.profile != target.profile:
                raise ValueError("Continued admission cannot replace its execution profile.")
        return self


class RebindInvocationCommand(_InvocationCommandModel):
    kind: Literal[InvocationLifecycleCommandKind.REBIND] = InvocationLifecycleCommandKind.REBIND
    expected_statuses: tuple[SessionStatus, ...]
    expected_run_epoch: StrictInt = Field(ge=0)
    expected_session_sha256: str
    expected_checkpoint_sha256: str
    expected_active_profile: ActiveInvocationExecutionProfile
    target_active_profile: ActiveInvocationExecutionProfile
    target_status: SessionStatus | None = None
    checkpoint_patch: InvocationCheckpointPatch = Field(default_factory=InvocationCheckpointPatch)

    @field_validator("expected_statuses", mode="before")
    @classmethod
    def copy_statuses(cls, value: object) -> tuple[SessionStatus, ...]:
        return AdmitInvocationCommand.copy_statuses(value)

    @field_validator("expected_session_sha256", "expected_checkpoint_sha256")
    @classmethod
    def validate_source_digest(cls, value: str, info) -> str:
        value = require_durable_clean_nonblank(value, info.field_name)
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"{info.field_name} must be lowercase SHA-256.")
        return value

    @model_validator(mode="after")
    def validate_rebind_authority(self) -> RebindInvocationCommand:
        expected = self.expected_active_profile
        target = self.target_active_profile
        if expected.session_id != self.session_id or target.session_id != self.session_id:
            raise ValueError("Rebind authority belongs to another session.")
        if expected.run_epoch not in {self.expected_run_epoch, self.expected_run_epoch - 1}:
            raise ValueError("Rebind source does not match the expected session epoch.")
        if target.run_epoch != self.expected_run_epoch + 1:
            raise ValueError("Rebind target must claim the next run epoch.")
        if target.interaction_id != expected.interaction_id or target.profile != expected.profile:
            raise ValueError("Rebind cannot replace interaction or profile authority.")
        if self.target_status is not None and self.target_status in {
            SessionStatus.COMPLETED,
            SessionStatus.FAILED,
        }:
            raise ValueError("Rebind cannot perform terminal settlement.")
        return self


class RejectInvocationCommand(_InvocationCommandModel):
    kind: Literal[InvocationLifecycleCommandKind.REJECT] = InvocationLifecycleCommandKind.REJECT
    expected_statuses: tuple[SessionStatus, ...]
    expected_run_epoch: StrictInt = Field(ge=0)
    expected_profile: ExecutionProfileIdentity
    candidate_profile: ExecutionProfileIdentity
    event: Event
    decision: ExecutionProfileDecision | None = None
    expected_active_profile: ActiveInvocationExecutionProfile | None = None

    @field_validator("expected_statuses", mode="before")
    @classmethod
    def copy_statuses(cls, value: object) -> tuple[SessionStatus, ...]:
        return AdmitInvocationCommand.copy_statuses(value)

    @field_validator("event", mode="before")
    @classmethod
    def copy_rejection_event(cls, value: Event) -> Event:
        return copy_event(value)

    @model_validator(mode="after")
    def validate_rejection_authority(self) -> RejectInvocationCommand:
        if self.event.session_id != self.session_id:
            raise ValueError("Rejection event belongs to another session.")
        changed = changed_execution_profile_components(
            self.expected_profile,
            self.candidate_profile,
        )
        if not changed:
            raise ValueError("Rejection requires a changed candidate profile.")
        if self.decision is not None:
            if self.decision.kind not in {
                ExecutionProfileDecisionKind.MIGRATION_REQUIRED,
                ExecutionProfileDecisionKind.REJECTED,
            }:
                raise ValueError("Only non-admitting decisions can reject an invocation.")
            if (
                self.decision.expected_profile != self.expected_profile
                or self.decision.candidate_profile != self.candidate_profile
                or self.decision.changed_component_classes != changed
                or self.decision.event != self.event
            ):
                raise ValueError("Rejection decision conflicts with command authority.")
        else:
            expected_payload = {
                "expected_profile_fingerprint": self.expected_profile.fingerprint,
                "candidate_profile_fingerprint": self.candidate_profile.fingerprint,
                "changed_component_classes": [component.value for component in changed],
            }
            if (
                self.event.type is not EventType.SESSION_EXECUTION_PROFILE_REJECTED
                or self.event.interaction_id is not None
                or self.event.payload != expected_payload
            ):
                raise ValueError("Rejection event conflicts with command authority.")
        if self.expected_active_profile is not None:
            if self.expected_active_profile.session_id != self.session_id:
                raise ValueError("Rejection authority belongs to another session.")
            if self.expected_active_profile.run_epoch not in {
                self.expected_run_epoch,
                self.expected_run_epoch - 1,
            }:
                raise ValueError("Rejection authority belongs to another run epoch.")
            if self.expected_active_profile.profile != self.expected_profile:
                raise ValueError("Rejection expectation conflicts with active authority.")
        return self


class SettleInvocationCommand(_InvocationCommandModel):
    kind: Literal[InvocationLifecycleCommandKind.SETTLE] = InvocationLifecycleCommandKind.SETTLE
    expected_run_epoch: StrictInt = Field(ge=1)
    expected_active_profile: ActiveInvocationExecutionProfile
    expected_authority_state: Literal["active", "released"] = "active"
    transition: InteractionTransitionSpec

    @model_validator(mode="after")
    def validate_settlement_authority(self) -> SettleInvocationCommand:
        active = self.expected_active_profile
        if active.session_id != self.session_id:
            raise ValueError("Settlement authority belongs to another session.")
        expected_profile_epoch = self.expected_run_epoch - (
            self.expected_authority_state == "released"
        )
        if active.run_epoch != expected_profile_epoch:
            raise ValueError("Settlement authority belongs to another run epoch.")
        if self.transition.event.session_id != self.session_id:
            raise ValueError("Settlement event belongs to another session.")
        interaction_id = self.transition.event.interaction_id
        if interaction_id is None or not all(
            event_envelope_authority_is_runtime_generated(
                self.transition.event,
                field_name=field_name,
                value=value,
            )
            for field_name, value in (
                ("session_id", self.session_id),
                ("interaction_id", interaction_id),
            )
        ):
            raise ValueError("Settlement event lacks runtime-owned session/interaction authority.")
        return self


class ReleaseInvocationCommand(_InvocationCommandModel):
    kind: Literal[InvocationLifecycleCommandKind.RELEASE] = InvocationLifecycleCommandKind.RELEASE
    expected_run_epoch: StrictInt = Field(ge=1)
    expected_active_profile: ActiveInvocationExecutionProfile
    settlement_transition: InteractionTransitionSpec | None = None
    recovery_claim_id: str | None = None
    terminal_session_event: Event | None = None
    _cleanup_authority: object | None = PrivateAttr(default=None)
    _store_authority: object | None = PrivateAttr(default=None)

    @field_validator("recovery_claim_id", mode="before")
    @classmethod
    def validate_recovery_claim_id(cls, value: object) -> str | None:
        if value is None:
            return None
        if type(value) is not str:
            raise TypeError("recovery_claim_id must be a string.")
        return require_durable_clean_nonblank(value, "recovery_claim_id")

    @field_validator("terminal_session_event", mode="before")
    @classmethod
    def copy_terminal_session_event(cls, value: Event | None) -> Event | None:
        return None if value is None else copy_event(value)

    @model_validator(mode="after")
    def validate_release_authority(self) -> ReleaseInvocationCommand:
        active = self.expected_active_profile
        if active.session_id != self.session_id:
            raise ValueError("Release authority belongs to another session.")
        if active.run_epoch != self.expected_run_epoch:
            raise ValueError("Release authority belongs to another run epoch.")
        authority_count = sum(
            value is not None
            for value in (
                self.settlement_transition,
                self.recovery_claim_id,
                self.terminal_session_event,
            )
        )
        if authority_count != 1:
            raise ValueError(
                "Release requires exactly one interaction settlement, terminal session "
                "event, or recovery claim."
            )
        if self.terminal_session_event is not None:
            if (
                self.terminal_session_event.session_id != self.session_id
                or self.terminal_session_event.interaction_id is not None
                or self.terminal_session_event.type
                not in {
                    EventType.SESSION_COMPLETED,
                    EventType.SESSION_FAILED,
                    EventType.SESSION_INTERRUPTED,
                }
            ):
                raise ValueError("Release terminal event must be session-scoped terminal evidence.")
            return self
        if self.settlement_transition is None:
            return self
        if self.settlement_transition.event.session_id != self.session_id:
            raise ValueError("Release settlement belongs to another session.")
        if self.settlement_transition.to_status not in {
            SessionStatus.COMPLETED,
            SessionStatus.FAILED,
            SessionStatus.INTERRUPTED,
        }:
            raise ValueError("Release requires a terminal settlement transition.")
        return self


InvocationLifecycleCommand: TypeAlias = Annotated[
    CreateInvocationCommand
    | AdmitInvocationCommand
    | RebindInvocationCommand
    | RejectInvocationCommand
    | SettleInvocationCommand
    | ReleaseInvocationCommand,
    Field(discriminator="kind"),
]
_INVOCATION_COMMAND_ADAPTER = TypeAdapter(InvocationLifecycleCommand)


@dataclass(frozen=True, slots=True)
class _ReleaseInvocationAuthority:
    token: object
    command_sha256: str


class _InvocationLifecycleCommandReceipt(BaseModel):
    """Durable evidence and original result for one atomic lifecycle mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    record_type: Literal["cayu.invocation-lifecycle-command-receipt"] = (
        _INVOCATION_LIFECYCLE_RECEIPT_RECORD_TYPE
    )
    schema_version: Literal[1] = _INVOCATION_LIFECYCLE_RECEIPT_SCHEMA_VERSION
    kind: InvocationLifecycleCommandKind
    command_identity: str
    command_sha256: str
    session_id: str
    session_instance_id: str
    result_session: Session
    active_profile: ActiveInvocationExecutionProfile
    record_sha256: str = ""

    @field_validator(
        "command_identity",
        "session_id",
        "command_sha256",
        "record_sha256",
    )
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        if info.field_name in {"command_sha256", "record_sha256"} and value == "":
            return value
        value = require_durable_clean_nonblank(value, info.field_name)
        if info.field_name.endswith("sha256") and (
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{info.field_name} must be lowercase SHA-256.")
        return value

    @field_validator("session_instance_id")
    @classmethod
    def validate_instance_id(cls, value: str) -> str:
        return SessionInvocationBinding.validate_session_instance_id(value)

    @field_validator("result_session", mode="before")
    @classmethod
    def copy_result_session(cls, value: object) -> Session:
        if type(value) is Session:
            return copy_session(value)
        return Session.model_validate(value)

    @model_validator(mode="after")
    def validate_record(self) -> _InvocationLifecycleCommandReceipt:
        if (
            self.result_session.id != self.session_id
            or self.result_session.instance_id != self.session_instance_id
            or self.active_profile.session_id != self.session_id
        ):
            raise ValueError("Lifecycle receipt result conflicts with its session authority.")
        expected_result_epoch = self.active_profile.run_epoch + (
            self.kind is InvocationLifecycleCommandKind.RELEASE
        )
        if self.result_session.run_epoch != expected_result_epoch:
            raise ValueError("Lifecycle receipt result conflicts with its run authority.")
        result_profile = execution_profile_from_session_metadata(self.result_session.metadata)
        if execution_profile_changes_authority(
            changed_execution_profile_components(result_profile, self.active_profile.profile)
        ):
            raise ValueError("Lifecycle receipt result metadata conflicts with its active profile.")
        result_ceiling = tool_capability_ceiling_from_session_metadata(self.result_session.metadata)
        ceiling_component = direct_tool_capability_ceiling_component(result_ceiling.tool_names)
        if result_profile.component(ceiling_component.component_class) != ceiling_component:
            raise ValueError("Lifecycle receipt result metadata conflicts with its tool ceiling.")
        provider_component = execution_profile_provider_target_component(
            self.result_session.provider_name,
            self.result_session.model,
        )
        if result_profile.component(provider_component.component_class) != provider_component:
            raise ValueError(
                "Lifecycle receipt result metadata conflicts with its provider target."
            )
        runtime_component = execution_profile_runtime_component(
            self.result_session.runtime_name,
            self.result_session.runtime_version,
        )
        if result_profile.component(runtime_component.component_class) != runtime_component:
            raise ValueError(
                "Lifecycle receipt result metadata conflicts with its runtime identity."
            )
        if (
            self.kind
            in {
                InvocationLifecycleCommandKind.CREATE,
                InvocationLifecycleCommandKind.ADMIT,
            }
            and self.result_session.status is not SessionStatus.RUNNING
        ):
            raise ValueError("Lifecycle admission receipt must return a running session.")
        if (
            self.kind is InvocationLifecycleCommandKind.RELEASE
            and self.result_session.status
            not in {
                SessionStatus.RUNNING,
                SessionStatus.COMPLETED,
                SessionStatus.FAILED,
                SessionStatus.INTERRUPTED,
            }
        ):
            raise ValueError("Lifecycle release receipt must return a settled session.")
        material = self.model_dump(mode="json", exclude={"record_sha256"})
        expected = sha256(
            canonical_bounded_durable_json_bytes(
                material,
                "invocation lifecycle receipt",
                max_bytes=INVOCATION_LIFECYCLE_RECEIPT_LEDGER_MAX_BYTES,
                max_nodes=_INVOCATION_LIFECYCLE_RECEIPT_LEDGER_MAX_NODES,
            )
        ).hexdigest()
        if self.record_sha256 not in {"", expected}:
            raise ValueError("Lifecycle receipt digest does not match its contents.")
        object.__setattr__(self, "record_sha256", expected)
        return self


def _projected_invocation_release_receipt(
    reserved_receipt: _InvocationLifecycleCommandReceipt,
    *,
    result_session: Session,
) -> _InvocationLifecycleCommandReceipt:
    """Build the largest release receipt possible for one active authority."""

    return _InvocationLifecycleCommandReceipt(
        kind=InvocationLifecycleCommandKind.RELEASE,
        command_identity=(
            f"{InvocationLifecycleCommandKind.RELEASE.value}:"
            f"{reserved_receipt.session_id}:{reserved_receipt.session_instance_id}:"
            f"{reserved_receipt.active_profile.run_epoch}"
        ),
        command_sha256="f" * 64,
        session_id=reserved_receipt.session_id,
        session_instance_id=reserved_receipt.session_instance_id,
        result_session=result_session.model_copy(
            update={
                "run_epoch": reserved_receipt.active_profile.run_epoch + 1,
                # This is the longest permitted status spelling, so the
                # projection cannot under-reserve the final receipt.
                "status": SessionStatus.INTERRUPTED,
            }
        ),
        active_profile=reserved_receipt.active_profile,
    )


def _require_projected_invocation_release_capacity(
    receipts: tuple[_InvocationLifecycleCommandReceipt, ...],
    reserved_receipt: _InvocationLifecycleCommandReceipt,
    *,
    result_session: Session,
) -> None:
    projected_release = _projected_invocation_release_receipt(
        reserved_receipt,
        result_session=result_session,
    )
    projected_receipts = {item.command_identity: item for item in receipts}
    projected_receipts[projected_release.command_identity] = projected_release
    projected_material = {
        "record_type": INVOCATION_LIFECYCLE_RECEIPT_LEDGER_RECORD_TYPE,
        "schema_version": INVOCATION_LIFECYCLE_RECEIPT_LEDGER_SCHEMA_VERSION,
        "receipts": [
            projected_receipts[key].model_dump(mode="json") for key in sorted(projected_receipts)
        ],
        "release_capacity_command_identity": None,
        # The final ledger digest is always a lowercase SHA-256 value.
        "record_sha256": "f" * 64,
    }
    canonical_bounded_durable_json_bytes(
        projected_material,
        "invocation lifecycle receipt ledger release capacity",
        max_bytes=INVOCATION_LIFECYCLE_RECEIPT_LEDGER_MAX_BYTES,
        max_nodes=_INVOCATION_LIFECYCLE_RECEIPT_LEDGER_MAX_NODES,
    )


class _InvocationLifecycleReceiptLedger(BaseModel):
    """Identity-keyed lifecycle receipts retained across later state changes."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    record_type: Literal["cayu.invocation-lifecycle-command-receipt-ledger"] = (
        INVOCATION_LIFECYCLE_RECEIPT_LEDGER_RECORD_TYPE
    )
    schema_version: Literal[1] = INVOCATION_LIFECYCLE_RECEIPT_LEDGER_SCHEMA_VERSION
    receipts: tuple[_InvocationLifecycleCommandReceipt, ...] = Field(
        default=(),
        max_length=INVOCATION_LIFECYCLE_RECEIPT_LEDGER_MAX_ITEMS,
    )
    release_capacity_command_identity: str | None = None
    record_sha256: str = ""

    @field_validator("receipts", mode="before")
    @classmethod
    def copy_receipts(cls, value: object) -> object:
        if type(value) not in (tuple, list):
            raise TypeError("Lifecycle receipt ledger entries must be a tuple or list.")
        receipt_values = cast("tuple[object, ...] | list[object]", value)
        if len(receipt_values) > INVOCATION_LIFECYCLE_RECEIPT_LEDGER_MAX_ITEMS:
            raise ValueError("Lifecycle receipt ledger exceeds its retained command limit.")
        return receipt_values

    @field_validator("release_capacity_command_identity")
    @classmethod
    def validate_release_capacity_identity(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_durable_clean_nonblank(value, "release_capacity_command_identity")

    @model_validator(mode="after")
    def validate_ledger(self) -> _InvocationLifecycleReceiptLedger:
        identities = tuple(item.command_identity for item in self.receipts)
        if tuple(sorted(identities)) != identities or len(set(identities)) != len(identities):
            raise ValueError("Lifecycle receipt ledger identities must be unique and sorted.")
        if self.receipts:
            create_receipts = tuple(
                item for item in self.receipts if item.kind is InvocationLifecycleCommandKind.CREATE
            )
            if len(create_receipts) > 1:
                raise ValueError("Lifecycle receipt ledger has conflicting creation authority.")
            baseline_authority = (
                create_receipts[0].active_profile.profile
                if create_receipts
                else execution_profile_baseline_from_session_metadata(
                    self.receipts[0].result_session.metadata
                )
            )
            if any(
                execution_profile_baseline_from_session_metadata(item.result_session.metadata)
                != baseline_authority
                for item in self.receipts
            ):
                raise ValueError("Lifecycle receipt ledger lost its immutable profile baseline.")
            active_receipts = tuple(
                item
                for item in self.receipts
                if item.kind
                in {
                    InvocationLifecycleCommandKind.CREATE,
                    InvocationLifecycleCommandKind.ADMIT,
                    InvocationLifecycleCommandKind.REBIND,
                }
            )
            active_epochs = tuple(item.active_profile.run_epoch for item in active_receipts)
            if len(set(active_epochs)) != len(active_epochs):
                raise ValueError(
                    "Lifecycle receipt ledger has conflicting active command authority."
                )
            if active_receipts:
                latest_active_receipt = max(
                    active_receipts,
                    key=lambda item: item.active_profile.run_epoch,
                )
                latest_release_identity = (
                    f"{InvocationLifecycleCommandKind.RELEASE.value}:"
                    f"{latest_active_receipt.session_id}:"
                    f"{latest_active_receipt.session_instance_id}:"
                    f"{latest_active_receipt.active_profile.run_epoch}"
                )
                latest_is_released = any(
                    item.command_identity == latest_release_identity for item in self.receipts
                )
                expected_reservation = (
                    None if latest_is_released else latest_active_receipt.command_identity
                )
                if self.release_capacity_command_identity != expected_reservation:
                    raise ValueError(
                        "Lifecycle receipt ledger lost its current release capacity authority."
                    )
        reserved_receipt = None
        if self.release_capacity_command_identity is not None:
            reserved_receipt = next(
                (
                    item
                    for item in self.receipts
                    if item.command_identity == self.release_capacity_command_identity
                ),
                None,
            )
            if reserved_receipt is None or reserved_receipt.kind not in {
                InvocationLifecycleCommandKind.CREATE,
                InvocationLifecycleCommandKind.ADMIT,
                InvocationLifecycleCommandKind.REBIND,
            }:
                raise ValueError(
                    "Lifecycle receipt ledger release capacity lacks active command authority."
                )
            projected_identity = (
                f"{InvocationLifecycleCommandKind.RELEASE.value}:"
                f"{reserved_receipt.session_id}:{reserved_receipt.session_instance_id}:"
                f"{reserved_receipt.active_profile.run_epoch}"
            )
            if any(item.command_identity == projected_identity for item in self.receipts):
                raise ValueError(
                    "Lifecycle receipt ledger retains release capacity after exact release."
                )
            if len(self.receipts) + 1 > INVOCATION_LIFECYCLE_RECEIPT_LEDGER_MAX_ITEMS:
                raise ValueError("Lifecycle receipt ledger exceeds its retained command limit.")
        material = self.model_dump(mode="json", exclude={"record_sha256"})
        expected = sha256(
            canonical_bounded_durable_json_bytes(
                material,
                "invocation lifecycle receipt ledger",
                max_bytes=INVOCATION_LIFECYCLE_RECEIPT_LEDGER_MAX_BYTES,
                max_nodes=_INVOCATION_LIFECYCLE_RECEIPT_LEDGER_MAX_NODES,
            )
        ).hexdigest()
        if self.record_sha256 not in {"", expected}:
            raise ValueError("Lifecycle receipt ledger digest does not match its contents.")
        object.__setattr__(self, "record_sha256", expected)
        canonical_bounded_durable_json_bytes(
            self.model_dump(mode="json"),
            "invocation lifecycle receipt ledger",
            max_bytes=INVOCATION_LIFECYCLE_RECEIPT_LEDGER_MAX_BYTES,
            max_nodes=_INVOCATION_LIFECYCLE_RECEIPT_LEDGER_MAX_NODES,
        )
        if reserved_receipt is not None:
            _require_projected_invocation_release_capacity(
                self.receipts,
                reserved_receipt,
                result_session=reserved_receipt.result_session,
            )
        return self


def _require_receipt_result_matches_live_session(
    receipt: _InvocationLifecycleCommandReceipt,
    session: Session,
) -> None:
    """Bind immutable receipt fields to the independently loaded durable session."""

    result = receipt.result_session
    if (
        result.id != session.id
        or result.instance_id != session.instance_id
        or result.agent_name != session.agent_name
        or result.parent_session_id != session.parent_session_id
        or result.causal_budget_id != session.causal_budget_id
        or result.runtime_name != session.runtime_name
        or result.runtime_version != session.runtime_version
        or result.environment_name != session.environment_name
        or result.created_at != session.created_at
        or result.invocation != session.invocation
        or execution_profile_baseline_from_session_metadata(result.metadata)
        != execution_profile_baseline_from_session_metadata(session.metadata)
    ):
        raise RuntimeError("Durable invocation lifecycle receipt forged session authority.")


def _require_receipt_result_matches_command(
    receipt: _InvocationLifecycleCommandReceipt,
    command: CreateInvocationCommand
    | AdmitInvocationCommand
    | RebindInvocationCommand
    | ReleaseInvocationCommand,
    session: Session,
) -> None:
    """Validate every command-derived result field before exact replay."""

    _require_receipt_result_matches_live_session(receipt, session)
    result = receipt.result_session
    if type(command) is CreateInvocationCommand:
        expected_causal_budget_id = command.request.causal_budget_id or command.session_id
        if (
            result.status is not SessionStatus.RUNNING
            or result.agent_name != command.request.agent_name
            or result.provider_name != command.identity.provider_name
            or result.model != command.identity.model
            or result.parent_session_id != command.request.parent_session_id
            or result.causal_budget_id != expected_causal_budget_id
            or result.runtime_name != command.identity.runtime_name
            or result.runtime_version != command.identity.runtime_version
            or result.environment_name != command.request.environment_name
            or result.labels != command.request.labels
            or session_user_metadata(result.metadata)
            != session_user_metadata(command.request.metadata)
            or tool_capability_ceiling_from_session_metadata(result.metadata)
            != command.request.tool_capability_ceiling
        ):
            raise RuntimeError("Durable invocation create receipt forged its result snapshot.")
        return
    if type(command) is AdmitInvocationCommand:
        if (
            result.status is not SessionStatus.RUNNING
            or tool_capability_ceiling_from_session_metadata(result.metadata)
            != command.tool_capability_ceiling
        ):
            raise RuntimeError("Durable invocation admission receipt forged its result snapshot.")
        return
    if type(command) is RebindInvocationCommand:
        expected_statuses = (
            command.expected_statuses if command.target_status is None else (command.target_status,)
        )
        if result.status not in expected_statuses:
            raise RuntimeError("Durable invocation rebind receipt forged its result status.")
        return
    if type(command) is not ReleaseInvocationCommand:
        raise AssertionError("Receipt validation received an unsupported command type.")
    release_command = command
    if release_command.settlement_transition is not None:
        transition = release_command.settlement_transition
        permitted_statuses = {transition.to_status}
        if transition.only_if_no_queued_messages:
            permitted_statuses.update(transition.from_statuses)
        if result.status not in permitted_statuses:
            raise RuntimeError("Durable invocation release receipt forged settlement status.")
        return
    if release_command.terminal_session_event is not None:
        expected_statuses_by_event_type: dict[str, SessionStatus] = {
            EventType.SESSION_COMPLETED: SessionStatus.COMPLETED,
            EventType.SESSION_FAILED: SessionStatus.FAILED,
            EventType.SESSION_INTERRUPTED: SessionStatus.INTERRUPTED,
        }
        expected_status = expected_statuses_by_event_type[
            release_command.terminal_session_event.type
        ]
        if result.status is not expected_status:
            raise RuntimeError("Durable invocation release receipt forged terminal status.")
        return
    if result.status not in {
        SessionStatus.COMPLETED,
        SessionStatus.FAILED,
        SessionStatus.INTERRUPTED,
    }:
        raise RuntimeError("Durable invocation recovery receipt forged terminal status.")


def _invocation_lifecycle_command_identity(command: InvocationLifecycleCommand) -> str:
    if type(command) is CreateInvocationCommand:
        epoch = command.active_profile.run_epoch
    elif type(command) in {AdmitInvocationCommand, RebindInvocationCommand}:
        profiled_command = cast(
            "AdmitInvocationCommand | RebindInvocationCommand",
            command,
        )
        epoch = profiled_command.target_active_profile.run_epoch
    elif type(command) is ReleaseInvocationCommand:
        epoch = command.expected_run_epoch
    else:
        raise TypeError("This lifecycle command does not own a command receipt.")
    return (
        f"{command.kind.value}:{command.session_id}:{command.expected_session_instance_id}:{epoch}"
    )


def _invocation_lifecycle_command_sha256(command: InvocationLifecycleCommand) -> str:
    material: dict[str, Any] = {
        "command": command.model_dump(mode="json", warnings=False),
    }
    if type(command) is CreateInvocationCommand:
        material["request_private_authority_sha256"] = (
            _run_request_invocation_lifecycle_authority_sha256(command.request)
        )
    return sha256(
        canonical_durable_json_bytes(
            material,
            "invocation lifecycle command",
        )
    ).hexdigest()


def _invocation_lifecycle_command_receipt(
    command: CreateInvocationCommand
    | AdmitInvocationCommand
    | RebindInvocationCommand
    | ReleaseInvocationCommand,
    *,
    active_profile: ActiveInvocationExecutionProfile,
    result_session: Session,
) -> _InvocationLifecycleCommandReceipt:
    return _InvocationLifecycleCommandReceipt(
        kind=command.kind,
        command_identity=_invocation_lifecycle_command_identity(command),
        command_sha256=_invocation_lifecycle_command_sha256(command),
        session_id=command.session_id,
        session_instance_id=command.expected_session_instance_id,
        result_session=result_session,
        active_profile=active_profile,
    )


def _invocation_lifecycle_receipt_ledger_from_checkpoint(
    checkpoint: dict[str, Any] | None,
) -> _InvocationLifecycleReceiptLedger:
    if checkpoint is None:
        return _InvocationLifecycleReceiptLedger()
    raw = checkpoint.get(INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY)
    if raw is None:
        return _InvocationLifecycleReceiptLedger()
    if (
        type(raw) is not dict
        or raw.get("record_type") != INVOCATION_LIFECYCLE_RECEIPT_LEDGER_RECORD_TYPE
        or raw.get("schema_version") != INVOCATION_LIFECYCLE_RECEIPT_LEDGER_SCHEMA_VERSION
        or "release_capacity_command_identity" not in raw
        or not raw.get("record_sha256")
    ):
        raise RuntimeError("Durable invocation lifecycle receipt ledger is incomplete.")
    raw_receipts = raw.get("receipts")
    if type(raw_receipts) is not list:
        raise RuntimeError("Durable invocation lifecycle receipt ledger is incomplete.")
    if len(raw_receipts) > INVOCATION_LIFECYCLE_RECEIPT_LEDGER_MAX_ITEMS:
        raise RuntimeError("Durable invocation lifecycle receipt ledger exceeds its limit.")
    if any(
        type(receipt) is not dict
        or receipt.get("record_type") != _INVOCATION_LIFECYCLE_RECEIPT_RECORD_TYPE
        or receipt.get("schema_version") != _INVOCATION_LIFECYCLE_RECEIPT_SCHEMA_VERSION
        or not receipt.get("record_sha256")
        for receipt in raw_receipts
    ):
        raise RuntimeError("Durable invocation lifecycle receipt ledger is incomplete.")
    try:
        canonical_bounded_durable_json_bytes(
            raw,
            "invocation lifecycle receipt ledger",
            max_bytes=INVOCATION_LIFECYCLE_RECEIPT_LEDGER_MAX_BYTES,
            max_nodes=_INVOCATION_LIFECYCLE_RECEIPT_LEDGER_MAX_NODES,
        )
        return _InvocationLifecycleReceiptLedger.model_validate(raw)
    except (DurableValueError, TypeError, ValueError) as exc:
        raise RuntimeError("Durable invocation lifecycle receipt ledger is malformed.") from exc


def invocation_lifecycle_receipt_history_present(
    checkpoint: dict[str, Any] | None,
) -> bool:
    """Return positive validated evidence that lifecycle authority once existed."""

    if checkpoint is None or INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY not in checkpoint:
        return False
    _invocation_lifecycle_receipt_ledger_from_checkpoint(checkpoint)
    return True


def require_invocation_lifecycle_release_capacity(
    checkpoint: dict[str, Any] | None,
    result_session: Session,
) -> None:
    """Keep mutable session fields inside the active release reservation."""

    ledger = _invocation_lifecycle_receipt_ledger_from_checkpoint(checkpoint)
    command_identity = ledger.release_capacity_command_identity
    if command_identity is None:
        return
    reserved_receipt = next(
        (item for item in ledger.receipts if item.command_identity == command_identity),
        None,
    )
    if reserved_receipt is None:  # pragma: no cover - ledger validation owns this
        raise RuntimeError("Invocation release capacity lost its active receipt.")
    if (
        result_session.id != reserved_receipt.session_id
        or result_session.instance_id != reserved_receipt.session_instance_id
        or result_session.run_epoch != reserved_receipt.active_profile.run_epoch
    ):
        raise SessionRunFenced("Session mutation lost active invocation release authority.")
    _require_projected_invocation_release_capacity(
        ledger.receipts,
        reserved_receipt,
        result_session=result_session,
    )


def _invocation_lifecycle_receipt_from_checkpoint(
    checkpoint: dict[str, Any] | None,
    *,
    command_identity: str,
) -> _InvocationLifecycleCommandReceipt | None:
    ledger = _invocation_lifecycle_receipt_ledger_from_checkpoint(checkpoint)
    return next(
        (item for item in ledger.receipts if item.command_identity == command_identity),
        None,
    )


def _receipt_ledger_material(
    receipts: tuple[_InvocationLifecycleCommandReceipt, ...],
    *,
    release_capacity_command_identity: str | None,
) -> dict[str, Any]:
    return {
        "record_type": INVOCATION_LIFECYCLE_RECEIPT_LEDGER_RECORD_TYPE,
        "schema_version": INVOCATION_LIFECYCLE_RECEIPT_LEDGER_SCHEMA_VERSION,
        "receipts": [item.model_dump(mode="json") for item in receipts],
        "release_capacity_command_identity": release_capacity_command_identity,
        "record_sha256": "f" * 64,
    }


def _compact_invocation_lifecycle_receipts(
    receipts: dict[str, _InvocationLifecycleCommandReceipt],
    *,
    retained_command_identity: str,
    release_capacity_command_identity: str | None,
    result_session: Session,
    enforce_encoded_limit: bool = False,
) -> tuple[_InvocationLifecycleCommandReceipt, ...]:
    """Retain the newest replay epochs while reserving the active release."""

    protected_identities = {retained_command_identity}
    if release_capacity_command_identity is not None:
        protected_identities.add(release_capacity_command_identity)
    item_limit = INVOCATION_LIFECYCLE_RECEIPT_LEDGER_MAX_ITEMS
    required_items = 1 + int(release_capacity_command_identity is not None)
    if item_limit < required_items:
        raise ValueError("Lifecycle receipt ledger cannot retain active release capacity.")

    def ordered() -> tuple[_InvocationLifecycleCommandReceipt, ...]:
        return tuple(receipts[key] for key in sorted(receipts))

    def fits(candidate: tuple[_InvocationLifecycleCommandReceipt, ...]) -> bool:
        if len(candidate) + int(release_capacity_command_identity is not None) > item_limit:
            return False
        if not enforce_encoded_limit:
            return True
        try:
            canonical_bounded_durable_json_bytes(
                _receipt_ledger_material(
                    candidate,
                    release_capacity_command_identity=release_capacity_command_identity,
                ),
                "invocation lifecycle receipt ledger",
                max_bytes=INVOCATION_LIFECYCLE_RECEIPT_LEDGER_MAX_BYTES,
                max_nodes=_INVOCATION_LIFECYCLE_RECEIPT_LEDGER_MAX_NODES,
            )
            if release_capacity_command_identity is not None:
                reserved_receipt = receipts[release_capacity_command_identity]
                _require_projected_invocation_release_capacity(
                    candidate,
                    reserved_receipt,
                    result_session=result_session,
                )
        except DurableValueError:
            return False
        return True

    candidate = ordered()
    while not fits(candidate):
        protected_epochs = {
            item.active_profile.run_epoch
            for item in candidate
            if item.command_identity in protected_identities
        }
        removable_epochs = sorted(
            {
                item.active_profile.run_epoch
                for item in candidate
                if item.active_profile.run_epoch not in protected_epochs
            }
        )
        if not removable_epochs:
            # The current command and its mandatory release projection do not
            # fit by themselves. Reject before the store mutates session state.
            raise ValueError("Lifecycle receipt ledger cannot retain active command authority.")
        oldest_epoch = removable_epochs[0]
        for identity, item in tuple(receipts.items()):
            if (
                identity not in protected_identities
                and item.active_profile.run_epoch == oldest_epoch
            ):
                del receipts[identity]
        candidate = ordered()
    return candidate


def checkpoint_with_invocation_lifecycle_receipt(
    checkpoint: dict[str, Any] | None,
    command: CreateInvocationCommand
    | AdmitInvocationCommand
    | RebindInvocationCommand
    | ReleaseInvocationCommand,
    *,
    active_profile: ActiveInvocationExecutionProfile,
    result_session: Session,
    _ledger: _InvocationLifecycleReceiptLedger | None = None,
) -> dict[str, Any]:
    updated = {} if checkpoint is None else copy_durable_json_object(checkpoint, "checkpoint")
    ledger = (
        _invocation_lifecycle_receipt_ledger_from_checkpoint(updated)
        if _ledger is None
        else _ledger
    )
    receipt = _invocation_lifecycle_command_receipt(
        command,
        active_profile=active_profile,
        result_session=result_session,
    )
    retained = {
        item.command_identity: item
        for item in ledger.receipts
        if item.command_identity != receipt.command_identity
    }
    retained[receipt.command_identity] = receipt
    release_capacity_command_identity = ledger.release_capacity_command_identity
    if command.kind in {
        InvocationLifecycleCommandKind.CREATE,
        InvocationLifecycleCommandKind.ADMIT,
        InvocationLifecycleCommandKind.REBIND,
    }:
        release_capacity_command_identity = receipt.command_identity
    elif command.kind is InvocationLifecycleCommandKind.RELEASE:
        if release_capacity_command_identity is not None:
            reserved_receipt = retained.get(release_capacity_command_identity)
            if reserved_receipt is None or (
                receipt.command_identity
                != (
                    f"{InvocationLifecycleCommandKind.RELEASE.value}:"
                    f"{reserved_receipt.session_id}:{reserved_receipt.session_instance_id}:"
                    f"{reserved_receipt.active_profile.run_epoch}"
                )
            ):
                raise RuntimeError(
                    "Invocation release conflicts with its retained receipt capacity."
                )
        release_capacity_command_identity = None
    compacted_receipts = _compact_invocation_lifecycle_receipts(
        retained,
        retained_command_identity=receipt.command_identity,
        release_capacity_command_identity=release_capacity_command_identity,
        result_session=result_session,
    )
    try:
        next_ledger = _InvocationLifecycleReceiptLedger(
            receipts=compacted_receipts,
            release_capacity_command_identity=release_capacity_command_identity,
        )
    except ValueError as error:
        if "encoded JSON" not in str(error):
            raise
        compacted_receipts = _compact_invocation_lifecycle_receipts(
            retained,
            retained_command_identity=receipt.command_identity,
            release_capacity_command_identity=release_capacity_command_identity,
            result_session=result_session,
            enforce_encoded_limit=True,
        )
        next_ledger = _InvocationLifecycleReceiptLedger(
            receipts=compacted_receipts,
            release_capacity_command_identity=release_capacity_command_identity,
        )
    updated[INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY] = next_ledger.model_dump(mode="json")
    return updated


def _require_release_authority(
    command: ReleaseInvocationCommand,
    *,
    token: object,
    attribute: str,
    message: str,
) -> None:
    authority = getattr(command, attribute)
    if (
        type(authority) is not _ReleaseInvocationAuthority
        or authority.token is not token
        or authority.command_sha256 != _invocation_lifecycle_command_sha256(command)
    ):
        raise SessionRunFenced(message)


def _release_invocation_command_with_cleanup_authority(
    command: ReleaseInvocationCommand,
) -> ReleaseInvocationCommand:
    """Mint process-local proof after the runtime has quiesced invocation work."""

    copied = copy_invocation_lifecycle_command(command)
    if type(copied) is not ReleaseInvocationCommand:
        raise TypeError("command must be a ReleaseInvocationCommand.")
    copied._cleanup_authority = _ReleaseInvocationAuthority(
        token=_RELEASE_CLEANUP_AUTHORITY_TOKEN,
        command_sha256=_invocation_lifecycle_command_sha256(copied),
    )
    return copied


async def _prepare_release_invocation_command_for_store(
    store: SessionStore,
    command: ReleaseInvocationCommand,
) -> ReleaseInvocationCommand:
    _require_release_authority(
        command,
        token=_RELEASE_CLEANUP_AUTHORITY_TOKEN,
        attribute="_cleanup_authority",
        message="Invocation cleanup has not proven quiescence for release.",
    )
    del store
    prepared = copy_invocation_lifecycle_command(command)
    assert type(prepared) is ReleaseInvocationCommand
    prepared._store_authority = _ReleaseInvocationAuthority(
        token=_RELEASE_STORE_AUTHORITY_TOKEN,
        command_sha256=_invocation_lifecycle_command_sha256(prepared),
    )
    return prepared


def require_invocation_release_store_authority(command: ReleaseInvocationCommand) -> None:
    """Reject direct store release calls that bypass cleanup and settlement proof."""

    _require_release_authority(
        command,
        token=_RELEASE_STORE_AUTHORITY_TOKEN,
        attribute="_store_authority",
        message="Invocation release lacks authenticated cleanup/settlement authority.",
    )


def invocation_release_replay_from_state(
    session: Session,
    checkpoint: dict[str, Any] | None,
    command: ReleaseInvocationCommand,
    *,
    _ledger: _InvocationLifecycleReceiptLedger | None = None,
) -> InvocationReleaseResult | None:
    """Validate an exact release receipt while the store owns its atomic state."""

    ledger = (
        _invocation_lifecycle_receipt_ledger_from_checkpoint(checkpoint)
        if _ledger is None
        else _ledger
    )
    receipt = next(
        (
            item
            for item in ledger.receipts
            if item.command_identity == _invocation_lifecycle_command_identity(command)
        ),
        None,
    )
    if receipt is None:
        return None
    if receipt.command_sha256 != _invocation_lifecycle_command_sha256(command):
        raise InvocationLifecycleCommandConflict(
            "Invocation release identity was reused with new authority."
        )
    if (
        receipt.kind is not InvocationLifecycleCommandKind.RELEASE
        or receipt.session_instance_id != command.expected_session_instance_id
        or receipt.active_profile != command.expected_active_profile
        or receipt.result_session.run_epoch != command.expected_run_epoch + 1
        or session.id != command.session_id
        or session.instance_id != command.expected_session_instance_id
        or session.run_epoch < receipt.result_session.run_epoch
    ):
        raise RuntimeError("Durable invocation release receipt conflicts with its session.")
    _require_receipt_result_matches_command(receipt, command, session)
    return InvocationReleaseResult(
        session=receipt.result_session,
        active_profile=receipt.active_profile,
        replayed=True,
    )


def require_released_invocation_command_authority(
    session: Session,
    checkpoint: dict[str, Any] | None,
    *,
    session_id: str,
    session_instance_id: str,
    active_profile: ActiveInvocationExecutionProfile,
    events: tuple[Event, ...] = (),
) -> None:
    """Require one exact, durably completed release at its successor epoch."""

    released_run_epoch = active_profile.run_epoch + 1
    require_invocation_command_authority(
        session,
        checkpoint,
        session_id=session_id,
        session_instance_id=session_instance_id,
        run_epochs=frozenset({released_run_epoch}),
        active_profile=active_profile,
        events=events,
    )
    receipt = _invocation_lifecycle_receipt_from_checkpoint(
        checkpoint,
        command_identity=(
            f"{InvocationLifecycleCommandKind.RELEASE.value}:"
            f"{session_id}:{session_instance_id}:{active_profile.run_epoch}"
        ),
    )
    if (
        receipt is None
        or receipt.kind is not InvocationLifecycleCommandKind.RELEASE
        or receipt.session_id != session_id
        or receipt.session_instance_id != session_instance_id
        or receipt.active_profile != active_profile
        or receipt.result_session.instance_id != session_instance_id
        or receipt.result_session.run_epoch != released_run_epoch
        or session.run_epoch != receipt.result_session.run_epoch
    ):
        raise SessionRunFenced("Invocation command lacks exact durable released-profile authority.")
    _require_receipt_result_matches_live_session(receipt, session)


def require_invocation_admission_source_authority(
    session: Session,
    checkpoint: dict[str, Any] | None,
    *,
    session_id: str,
    session_instance_id: str,
    expected_active_profile: ActiveInvocationExecutionProfile | None,
) -> None:
    """Require a virgin session or one exact typed-release predecessor."""

    current_active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
    if expected_active_profile is None:
        if current_active_profile is not None or invocation_lifecycle_receipt_history_present(
            checkpoint
        ):
            raise SessionRunFenced(
                "Invocation admission lacks exact predecessor release authority."
            )
        return
    require_released_invocation_command_authority(
        session,
        checkpoint,
        session_id=session_id,
        session_instance_id=session_instance_id,
        active_profile=expected_active_profile,
    )


class InvocationMutationResult(BaseModel):
    """One admitted or rebound invocation and its exact active authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session: Session
    active_profile: ActiveInvocationExecutionProfile
    replayed: StrictBool = False

    @field_validator("session", mode="before")
    @classmethod
    def copy_result_session(cls, value: Session) -> Session:
        return copy_session(value)

    @model_validator(mode="after")
    def validate_result_authority(self) -> InvocationMutationResult:
        if self.active_profile.session_id != self.session.id:
            raise ValueError("Invocation result authority belongs to another session.")
        if self.active_profile.run_epoch != self.session.run_epoch:
            raise ValueError("Invocation result authority belongs to another run epoch.")
        return self


class InvocationReleaseResult(BaseModel):
    """Exact durable result of releasing one invocation run epoch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session: Session
    active_profile: ActiveInvocationExecutionProfile
    replayed: StrictBool

    @field_validator("session", mode="before")
    @classmethod
    def copy_result_session(cls, value: Session) -> Session:
        return copy_session(value)

    @model_validator(mode="after")
    def validate_release_result(self) -> InvocationReleaseResult:
        if self.active_profile.session_id != self.session.id:
            raise ValueError("Release result authority belongs to another session.")
        if self.session.run_epoch != self.active_profile.run_epoch + 1:
            raise ValueError("Release result does not carry the fenced successor epoch.")
        if self.session.status not in {
            SessionStatus.RUNNING,
            SessionStatus.COMPLETED,
            SessionStatus.FAILED,
            SessionStatus.INTERRUPTED,
        }:
            raise ValueError("Release result does not carry a settled invocation state.")
        return self


InvocationLifecycleResult: TypeAlias = (
    InvocationMutationResult
    | InvocationReleaseResult
    | ExecutionProfileRejectionResult
    | InteractionTransitionResult
)


def copy_invocation_lifecycle_command(command: object) -> InvocationLifecycleCommand:
    """Revalidate and detach one command before any store receives it."""

    copied = _INVOCATION_COMMAND_ADAPTER.validate_python(command)
    if type(command) is ReleaseInvocationCommand and type(copied) is ReleaseInvocationCommand:
        digest = _invocation_lifecycle_command_sha256(copied)
        for attribute, token in (
            ("_cleanup_authority", _RELEASE_CLEANUP_AUTHORITY_TOKEN),
            ("_store_authority", _RELEASE_STORE_AUTHORITY_TOKEN),
        ):
            authority = getattr(command, attribute)
            if (
                type(authority) is _ReleaseInvocationAuthority
                and authority.token is token
                and authority.command_sha256 == digest
            ):
                setattr(
                    copied,
                    attribute,
                    _ReleaseInvocationAuthority(token=token, command_sha256=digest),
                )
    return copied


async def _replay_invocation_lifecycle_command(
    store: SessionStore,
    command: InvocationLifecycleCommand,
) -> InvocationMutationResult | InvocationReleaseResult | None:
    if type(command) not in {
        CreateInvocationCommand,
        AdmitInvocationCommand,
        RebindInvocationCommand,
        ReleaseInvocationCommand,
    }:
        return None
    replay_command = cast(
        "CreateInvocationCommand | AdmitInvocationCommand | RebindInvocationCommand | ReleaseInvocationCommand",
        command,
    )
    session = await store.load(replay_command.session_id)
    if session is None:
        return None
    checkpoint = await store.load_checkpoint(replay_command.session_id)
    identity = _invocation_lifecycle_command_identity(replay_command)
    receipt = _invocation_lifecycle_receipt_from_checkpoint(
        checkpoint,
        command_identity=identity,
    )
    if receipt is None:
        return None
    if receipt.command_sha256 != _invocation_lifecycle_command_sha256(replay_command):
        raise InvocationLifecycleCommandConflict(
            "Invocation lifecycle command identity was reused with new authority."
        )
    if (
        receipt.kind is not replay_command.kind
        or receipt.session_id != replay_command.session_id
        or receipt.session_instance_id != replay_command.expected_session_instance_id
        or session.instance_id != replay_command.expected_session_instance_id
    ):
        raise RuntimeError("Durable invocation lifecycle receipt conflicts with its session.")
    if session.run_epoch < receipt.result_session.run_epoch:
        raise RuntimeError("Durable invocation lifecycle receipt is ahead of its session.")
    if type(replay_command) is CreateInvocationCommand:
        expected_active = replay_command.active_profile
    elif type(replay_command) in {AdmitInvocationCommand, RebindInvocationCommand}:
        profiled_command = cast(
            "AdmitInvocationCommand | RebindInvocationCommand",
            replay_command,
        )
        expected_active = profiled_command.target_active_profile
    elif type(replay_command) is ReleaseInvocationCommand:
        expected_active = replay_command.expected_active_profile
    else:
        raise AssertionError("Replay validation received an unsupported command type.")
    if receipt.active_profile != expected_active:
        raise RuntimeError("Durable invocation lifecycle receipt lost active authority.")
    _require_receipt_result_matches_command(receipt, replay_command, session)
    if type(replay_command) is ReleaseInvocationCommand:
        return InvocationReleaseResult(
            session=receipt.result_session,
            active_profile=receipt.active_profile,
            replayed=True,
        )
    return InvocationMutationResult(
        session=receipt.result_session,
        active_profile=receipt.active_profile,
        replayed=True,
    )


def _apply_checkpoint_patch(
    patch: InvocationCheckpointPatch,
    checkpoint: dict[str, Any] | None,
) -> dict[str, Any] | None:
    return apply_runtime_publication_checkpoint_mutation(patch.mutation, checkpoint)


def _invocation_session_state_sha256(session: Session) -> str:
    return sha256(
        canonical_durable_json_bytes(
            session.model_dump(mode="json"),
            "invocation lifecycle source session",
        )
    ).hexdigest()


def invocation_checkpoint_state_sha256(
    checkpoint: dict[str, Any] | None,
) -> str:
    return sha256(
        canonical_durable_json_bytes(
            checkpoint,
            "invocation lifecycle source checkpoint",
        )
    ).hexdigest()


def require_invocation_command_authority(
    session: Session,
    checkpoint: dict[str, Any] | None,
    *,
    session_id: str,
    session_instance_id: str,
    run_epochs: frozenset[int],
    active_profile: ActiveInvocationExecutionProfile | None,
    events: tuple[Event, ...] = (),
) -> None:
    if session.id != session_id or session.instance_id != session_instance_id:
        raise SessionRunFenced("Invocation command belongs to another session incarnation.")
    if session.run_epoch not in run_epochs:
        rendered = ", ".join(str(item) for item in sorted(run_epochs))
        raise SessionRunFenced(
            "Invocation command lost its run epoch: expected one of "
            f"{{{rendered}}}, current {session.run_epoch}."
        )
    if active_profile is not None:
        current = active_invocation_execution_profile_from_checkpoint(checkpoint)
        if current != active_profile:
            raise SessionRunFenced("Invocation command lost its active profile authority.")
    if any(
        event.session_id != session.id
        or event.agent_name != session.agent_name
        or event.environment_name != session.environment_name
        for event in events
    ):
        raise SessionRunFenced("Invocation command event conflicts with session authority.")


def prepare_rebind_invocation_command(
    session: Session,
    checkpoint: dict[str, Any] | None,
    *,
    expected_statuses: set[SessionStatus] | frozenset[SessionStatus],
    checkpoint_transform: Callable[
        [Session, dict[str, Any] | None],
        dict[str, Any] | None,
    ],
    target_status: SessionStatus | None = None,
) -> RebindInvocationCommand:
    """Prepare one exact recovery/rebind command from an authenticated snapshot.

    The caller may calculate ordinary checkpoint state, but this factory keeps
    lifecycle authority out of the resulting generic CAS patch.  The command
    adapter owns the epoch/profile transition and rejects a stale snapshot.
    """

    if type(session) is not Session:
        raise TypeError("session must be a Session.")
    if not callable(checkpoint_transform):
        raise TypeError("checkpoint_transform must be callable.")
    source_checkpoint = (
        None if checkpoint is None else copy_durable_json_object(checkpoint, "checkpoint")
    )
    expected_profile = active_invocation_execution_profile_from_checkpoint(source_checkpoint)
    if type(expected_profile) is not ActiveInvocationExecutionProfile:
        raise SessionRunFenced("Invocation rebind requires active profile authority.")
    if expected_profile.session_id != session.id or expected_profile.run_epoch not in {
        session.run_epoch,
        session.run_epoch - 1,
    }:
        raise SessionRunFenced("Invocation rebind snapshot conflicts with the session epoch.")
    transformed = checkpoint_transform(
        copy_session(session),
        (
            None
            if source_checkpoint is None
            else copy_durable_json_object(source_checkpoint, "checkpoint")
        ),
    )
    desired_checkpoint = (
        None if transformed is None else copy_durable_json_object(transformed, "checkpoint")
    )
    target_profile = active_invocation_execution_profile_from_checkpoint(desired_checkpoint)
    if type(target_profile) is not ActiveInvocationExecutionProfile:
        raise ValueError("Invocation rebind transform removed active profile authority.")
    if (
        target_profile.session_id != session.id
        or target_profile.interaction_id != expected_profile.interaction_id
        or target_profile.profile != expected_profile.profile
        or target_profile.run_epoch != session.run_epoch + 1
    ):
        raise ValueError("Invocation rebind transform produced conflicting authority.")

    for authority_key in (
        CHECKPOINT_SCHEMA_VERSION_KEY,
        INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
    ):
        source_value = None if source_checkpoint is None else source_checkpoint.get(authority_key)
        desired_value = (
            None if desired_checkpoint is None else desired_checkpoint.get(authority_key)
        )
        if canonical_durable_json_bytes(
            source_value,
            f"checkpoint.{authority_key}",
        ) != canonical_durable_json_bytes(
            desired_value,
            f"desired_checkpoint.{authority_key}",
        ):
            raise ValueError("Invocation rebind transform attempted to mutate lifecycle authority.")

    mutation = runtime_publication_checkpoint_mutation(
        source_checkpoint,
        desired_checkpoint,
    )
    ordinary_mutation = RuntimePublicationMutation(
        operations=tuple(
            operation
            for operation in mutation.operations
            if operation.key != ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY
        )
    )
    return RebindInvocationCommand(
        session_id=session.id,
        expected_session_instance_id=session.instance_id,
        expected_statuses=tuple(sorted(expected_statuses, key=str)),
        expected_run_epoch=session.run_epoch,
        expected_session_sha256=_invocation_session_state_sha256(session),
        expected_checkpoint_sha256=invocation_checkpoint_state_sha256(source_checkpoint),
        expected_active_profile=expected_profile,
        target_active_profile=target_profile,
        target_status=target_status,
        checkpoint_patch=InvocationCheckpointPatch(mutation=ordinary_mutation),
    )


async def apply_invocation_lifecycle_command(
    store: SessionStore,
    command: InvocationLifecycleCommand,
) -> InvocationLifecycleResult:
    """Apply one version-1 command through a store's atomic lifecycle primitives."""

    copied = copy_invocation_lifecycle_command(command)
    if type(copied) is ReleaseInvocationCommand:
        copied = await _prepare_release_invocation_command_for_store(store, copied)
    else:
        replay = await _replay_invocation_lifecycle_command(store, copied)
        if replay is not None:
            return replay
    if type(copied) is CreateInvocationCommand:

        def create_checkpoint(session: Session, checkpoint: dict[str, Any] | None):
            if session.instance_id != copied.expected_session_instance_id:
                raise SessionRunFenced("Create command received another session incarnation.")
            updated = _apply_checkpoint_patch(copied.checkpoint_patch, checkpoint)
            updated = checkpoint_with_active_invocation_execution_profile(
                updated,
                session_id=copied.active_profile.session_id,
                interaction_id=copied.active_profile.interaction_id,
                run_epoch=copied.active_profile.run_epoch,
                profile=copied.active_profile.profile,
            )
            return updated

        def record_create_result(session: Session, checkpoint: dict[str, Any] | None):
            return checkpoint_with_invocation_lifecycle_receipt(
                checkpoint,
                copied,
                active_profile=copied.active_profile,
                result_session=session,
            )

        def initialize_operations(_session: Session) -> dict[str, dict[str, Any]]:
            if copied.tool_discovery_initialization is None:
                return {}
            if (
                _session.id != copied.session_id
                or _session.agent_name != copied.tool_discovery_initialization.agent_name
            ):
                raise SessionRunFenced(
                    "Create invocation discovery initialization belongs to another session."
                )
            return initial_tool_discovery_operation_records_from_initialization(
                copied.tool_discovery_initialization,
                root_invocation_id=_session.invocation.root_invocation_id,
            )

        try:
            with _invocation_lifecycle_authority_mutation_scope():
                session = await store.create(
                    copied.request,
                    identity=copied.identity,
                    interaction_started_event=copied.interaction_started_event,
                    interaction_source_messages=list(copied.interaction_source_messages),
                    checkpoint_transform=create_checkpoint,
                    result_checkpoint_transform=record_create_result,
                    operation_initializer=initialize_operations,
                )
        except Exception:
            replay = await _replay_invocation_lifecycle_command(store, copied)
            if replay is not None:
                return replay
            raise
        return InvocationMutationResult(
            session=session,
            active_profile=copied.active_profile,
        )

    if type(copied) is AdmitInvocationCommand:

        def admit_checkpoint(session: Session, checkpoint: dict[str, Any] | None):
            require_invocation_admission_source_authority(
                session,
                checkpoint,
                session_id=copied.session_id,
                session_instance_id=copied.expected_session_instance_id,
                expected_active_profile=copied.expected_active_profile,
            )
            require_invocation_command_authority(
                session,
                checkpoint,
                session_id=copied.session_id,
                session_instance_id=copied.expected_session_instance_id,
                run_epochs=frozenset({copied.expected_run_epoch}),
                active_profile=copied.expected_active_profile,
                events=(
                    ()
                    if copied.interaction_started_event is None
                    else (copied.interaction_started_event,)
                ),
            )
            if invocation_checkpoint_state_sha256(checkpoint) != copied.expected_checkpoint_sha256:
                raise SessionRunFenced(
                    "Invocation admission source checkpoint changed after command preparation."
                )
            return _apply_checkpoint_patch(copied.checkpoint_patch, checkpoint)

        def record_admit_result(session: Session, checkpoint: dict[str, Any] | None):
            return checkpoint_with_invocation_lifecycle_receipt(
                checkpoint,
                copied,
                active_profile=copied.target_active_profile,
                result_session=session,
            )

        try:
            with _invocation_lifecycle_authority_mutation_scope():
                session = await store.admit_session_invocation(
                    copied.session_id,
                    admission=SessionInvocationAdmission(
                        from_statuses=frozenset(copied.expected_statuses),
                        checkpoint_transform=admit_checkpoint,
                        result_checkpoint_transform=record_admit_result,
                        execution_profile=copied.target_active_profile.profile,
                        interaction_source_messages=copied.interaction_source_messages,
                        tool_capability_ceiling=copied.tool_capability_ceiling,
                        interaction_started_event=copied.interaction_started_event,
                        continued_interaction_id=copied.continued_interaction_id,
                        defer_interaction_source=copied.defer_interaction_source,
                        model_transition=copied.model_transition,
                        execution_profile_decision=copied.execution_profile_decision,
                        expected_active_invocation_profile=copied.expected_active_profile,
                        allow_pending_initial_interaction=(
                            copied.allow_pending_initial_interaction
                        ),
                    ),
                )
        except Exception:
            replay = await _replay_invocation_lifecycle_command(store, copied)
            if replay is not None:
                return replay
            raise
        return InvocationMutationResult(
            session=session,
            active_profile=copied.target_active_profile,
        )

    if type(copied) is RebindInvocationCommand:

        def rebind_checkpoint(session: Session, checkpoint: dict[str, Any] | None):
            require_invocation_command_authority(
                session,
                checkpoint,
                session_id=copied.session_id,
                session_instance_id=copied.expected_session_instance_id,
                run_epochs=frozenset({copied.expected_run_epoch}),
                active_profile=copied.expected_active_profile,
            )
            if (
                _invocation_session_state_sha256(session) != copied.expected_session_sha256
                or invocation_checkpoint_state_sha256(checkpoint)
                != copied.expected_checkpoint_sha256
            ):
                raise SessionRunFenced(
                    "Invocation rebind source state changed after command preparation."
                )
            updated = _apply_checkpoint_patch(copied.checkpoint_patch, checkpoint)
            updated = checkpoint_with_active_invocation_execution_profile(
                updated,
                session_id=copied.target_active_profile.session_id,
                interaction_id=copied.target_active_profile.interaction_id,
                run_epoch=copied.target_active_profile.run_epoch,
                profile=copied.target_active_profile.profile,
                expected=copied.expected_active_profile,
            )
            return updated

        def record_rebind_result(session: Session, checkpoint: dict[str, Any] | None):
            return checkpoint_with_invocation_lifecycle_receipt(
                checkpoint,
                copied,
                active_profile=copied.target_active_profile,
                result_session=session,
            )

        try:
            with _invocation_lifecycle_authority_mutation_scope():
                if copied.target_status is None:
                    session = await store.fence_run_and_transform_checkpoint(
                        copied.session_id,
                        statuses=set(copied.expected_statuses),
                        checkpoint_transform=rebind_checkpoint,
                        result_checkpoint_transform=record_rebind_result,
                    )
                else:
                    session = await store.transition_status_and_checkpoint(
                        copied.session_id,
                        from_statuses=set(copied.expected_statuses),
                        to_status=copied.target_status,
                        checkpoint_transform=rebind_checkpoint,
                        result_checkpoint_transform=record_rebind_result,
                    )
        except Exception:
            replay = await _replay_invocation_lifecycle_command(store, copied)
            if replay is not None:
                return replay
            raise
        return InvocationMutationResult(
            session=session,
            active_profile=copied.target_active_profile,
        )

    if type(copied) is RejectInvocationCommand:
        if copied.expected_active_profile is not None:
            return await store.reject_active_invocation_execution_profile(
                copied.session_id,
                expected_session_instance_id=copied.expected_session_instance_id,
                expected_statuses=set(copied.expected_statuses),
                expected_run_epoch=copied.expected_run_epoch,
                expected_active_invocation_profile=copied.expected_active_profile,
                candidate_profile=copied.candidate_profile,
                event=copied.event,
                decision=copied.decision,
            )
        return await store.reject_execution_profile_resume(
            copied.session_id,
            expected_session_instance_id=copied.expected_session_instance_id,
            expected_statuses=set(copied.expected_statuses),
            expected_run_epoch=copied.expected_run_epoch,
            expected_profile=copied.expected_profile,
            candidate_profile=copied.candidate_profile,
            event=copied.event,
            decision=copied.decision,
        )

    if type(copied) is SettleInvocationCommand:
        return await store.settle_session_invocation(copied)

    if type(copied) is ReleaseInvocationCommand:
        try:
            return await store.release_session_invocation(copied)
        except Exception:
            replay = await _replay_invocation_lifecycle_command(store, copied)
            if replay is not None:
                return replay
            raise

    raise AssertionError("Invocation command validation returned an unknown command type.")


__all__ = [
    "INVOCATION_LIFECYCLE_COMMAND_VERSION",
    "AdmitInvocationCommand",
    "AdmittedInvocationBinding",
    "CreateInvocationCommand",
    "InvocationCheckpointPatch",
    "InvocationContext",
    "InvocationLifecycleCommand",
    "InvocationLifecycleCommandConflict",
    "InvocationLifecycleCommandKind",
    "InvocationLifecycleResult",
    "InvocationMutationResult",
    "InvocationReleaseResult",
    "PreparedInvocationBinding",
    "RebindInvocationCommand",
    "RejectInvocationCommand",
    "ReleaseInvocationCommand",
    "SettleInvocationCommand",
    "apply_invocation_lifecycle_command",
    "copy_invocation_lifecycle_command",
    "invocation_checkpoint_state_sha256",
    "invocation_lifecycle_receipt_history_present",
    "prepare_rebind_invocation_command",
    "require_invocation_admission_source_authority",
    "require_invocation_command_authority",
]
