"""Live invocation context, cleanup admission and asynchronous command dispatch."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError
from datetime import datetime
from hashlib import sha256
from typing import Any, Never, SupportsIndex, cast

from cayu._validation import (
    canonical_durable_json_bytes,
    copy_durable_json_object,
    require_durable_clean_nonblank,
)
from cayu.budgets.base import BudgetPolicy
from cayu.execution_profiles import (
    ExecutionProfileIdentity,
    direct_tool_capability_ceiling_component,
    execution_profile_provider_target_component,
)
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime._work_attempt_invocation import WorkAttemptInvocationAuthority
from cayu.runtime.invocation_release import InvocationReleaseEvidence
from cayu.runtime.loop_policies import LoopPolicy
from cayu.sessions._checkpoint_preservation import _invocation_lifecycle_authority_mutation_scope
from cayu.sessions._execution_profile_checkpoint import (
    ActiveInvocationExecutionProfile,
    active_invocation_execution_profile_from_checkpoint,
    checkpoint_with_active_invocation_execution_profile,
)
from cayu.sessions._invocation_lifecycle import (
    _INVOCATION_COMMAND_ADAPTER as _INVOCATION_COMMAND_ADAPTER,
)
from cayu.sessions._invocation_lifecycle import (
    _INVOCATION_LIFECYCLE_RECEIPT_LEDGER_MAX_NODES as _INVOCATION_LIFECYCLE_RECEIPT_LEDGER_MAX_NODES,
)
from cayu.sessions._invocation_lifecycle import (
    _INVOCATION_LIFECYCLE_RECEIPT_RECORD_TYPE as _INVOCATION_LIFECYCLE_RECEIPT_RECORD_TYPE,
)
from cayu.sessions._invocation_lifecycle import (
    _INVOCATION_LIFECYCLE_RECEIPT_SCHEMA_VERSION as _INVOCATION_LIFECYCLE_RECEIPT_SCHEMA_VERSION,
)
from cayu.sessions._invocation_lifecycle import (
    _RELEASE_CLEANUP_AUTHORITY_TOKEN as _RELEASE_CLEANUP_AUTHORITY_TOKEN,
)
from cayu.sessions._invocation_lifecycle import (
    _RELEASE_STORE_AUTHORITY_TOKEN as _RELEASE_STORE_AUTHORITY_TOKEN,
)
from cayu.sessions._invocation_lifecycle import (
    INVOCATION_LIFECYCLE_COMMAND_VERSION as INVOCATION_LIFECYCLE_COMMAND_VERSION,
)
from cayu.sessions._invocation_lifecycle import (
    INVOCATION_LIFECYCLE_RECEIPT_LEDGER_MAX_BYTES as INVOCATION_LIFECYCLE_RECEIPT_LEDGER_MAX_BYTES,
)
from cayu.sessions._invocation_lifecycle import (
    INVOCATION_LIFECYCLE_RECEIPT_LEDGER_MAX_ITEMS as INVOCATION_LIFECYCLE_RECEIPT_LEDGER_MAX_ITEMS,
)

# Keep established imports and historical pickle paths bound to the canonical objects.
from cayu.sessions._invocation_lifecycle import AdmitInvocationCommand as AdmitInvocationCommand
from cayu.sessions._invocation_lifecycle import (
    AdmittedInvocationBinding as AdmittedInvocationBinding,
)
from cayu.sessions._invocation_lifecycle import CreateInvocationCommand as CreateInvocationCommand
from cayu.sessions._invocation_lifecycle import InvocationBinding as InvocationBinding
from cayu.sessions._invocation_lifecycle import (
    InvocationCheckpointPatch as InvocationCheckpointPatch,
)
from cayu.sessions._invocation_lifecycle import (
    InvocationLifecycleCommand as InvocationLifecycleCommand,
)
from cayu.sessions._invocation_lifecycle import (
    InvocationLifecycleCommandConflict as InvocationLifecycleCommandConflict,
)
from cayu.sessions._invocation_lifecycle import (
    InvocationLifecycleCommandKind as InvocationLifecycleCommandKind,
)
from cayu.sessions._invocation_lifecycle import (
    InvocationLifecycleResult as InvocationLifecycleResult,
)
from cayu.sessions._invocation_lifecycle import InvocationMutationResult as InvocationMutationResult
from cayu.sessions._invocation_lifecycle import InvocationReleaseResult as InvocationReleaseResult
from cayu.sessions._invocation_lifecycle import (
    PreparedInvocationBinding as PreparedInvocationBinding,
)
from cayu.sessions._invocation_lifecycle import RebindInvocationCommand as RebindInvocationCommand
from cayu.sessions._invocation_lifecycle import RejectInvocationCommand as RejectInvocationCommand
from cayu.sessions._invocation_lifecycle import ReleaseInvocationCommand as ReleaseInvocationCommand
from cayu.sessions._invocation_lifecycle import SettleInvocationCommand as SettleInvocationCommand
from cayu.sessions._invocation_lifecycle import _apply_checkpoint_patch as _apply_checkpoint_patch
from cayu.sessions._invocation_lifecycle import (
    _checkpoint_after_predecessor_terminal_decision as _checkpoint_after_predecessor_terminal_decision,
)
from cayu.sessions._invocation_lifecycle import (
    _compact_invocation_lifecycle_receipts as _compact_invocation_lifecycle_receipts,
)
from cayu.sessions._invocation_lifecycle import (
    _ExternalExecutionOrigin as _ExternalExecutionOrigin,
)
from cayu.sessions._invocation_lifecycle import (
    _invocation_lifecycle_command_identity as _invocation_lifecycle_command_identity,
)
from cayu.sessions._invocation_lifecycle import (
    _invocation_lifecycle_command_receipt as _invocation_lifecycle_command_receipt,
)
from cayu.sessions._invocation_lifecycle import (
    _invocation_lifecycle_command_sha256 as _invocation_lifecycle_command_sha256,
)
from cayu.sessions._invocation_lifecycle import (
    _invocation_lifecycle_receipt_from_checkpoint as _invocation_lifecycle_receipt_from_checkpoint,
)
from cayu.sessions._invocation_lifecycle import (
    _invocation_lifecycle_receipt_ledger_from_checkpoint as _invocation_lifecycle_receipt_ledger_from_checkpoint,
)
from cayu.sessions._invocation_lifecycle import (
    _invocation_session_state_sha256 as _invocation_session_state_sha256,
)
from cayu.sessions._invocation_lifecycle import _InvocationCommandModel as _InvocationCommandModel
from cayu.sessions._invocation_lifecycle import (
    _InvocationLifecycleCommandReceipt as _InvocationLifecycleCommandReceipt,
)
from cayu.sessions._invocation_lifecycle import (
    _InvocationLifecycleReceiptLedger as _InvocationLifecycleReceiptLedger,
)
from cayu.sessions._invocation_lifecycle import (
    _projected_invocation_release_receipt as _projected_invocation_release_receipt,
)
from cayu.sessions._invocation_lifecycle import _receipt_ledger_material as _receipt_ledger_material
from cayu.sessions._invocation_lifecycle import (
    _ReleaseInvocationAuthority as _ReleaseInvocationAuthority,
)
from cayu.sessions._invocation_lifecycle import (
    _require_projected_invocation_release_capacity as _require_projected_invocation_release_capacity,
)
from cayu.sessions._invocation_lifecycle import (
    _require_receipt_result_matches_command as _require_receipt_result_matches_command,
)
from cayu.sessions._invocation_lifecycle import (
    _require_receipt_result_matches_live_session as _require_receipt_result_matches_live_session,
)
from cayu.sessions._invocation_lifecycle import (
    _require_release_authority as _require_release_authority,
)
from cayu.sessions._invocation_lifecycle import (
    _require_released_invocation_command_receipt as _require_released_invocation_command_receipt,
)
from cayu.sessions._invocation_lifecycle import (
    _validate_invocation_binding as _validate_invocation_binding,
)
from cayu.sessions._invocation_lifecycle import (
    copy_invocation_lifecycle_command as copy_invocation_lifecycle_command,
)
from cayu.sessions._invocation_lifecycle import (
    invocation_admission_command_sha256 as invocation_admission_command_sha256,
)
from cayu.sessions._invocation_lifecycle import (
    invocation_lifecycle_receipt_history_present as invocation_lifecycle_receipt_history_present,
)
from cayu.sessions._invocation_lifecycle import (
    invocation_release_replay_from_state as invocation_release_replay_from_state,
)
from cayu.sessions._invocation_lifecycle import (
    reconcile_invocation_admission_from_state as reconcile_invocation_admission_from_state,
)
from cayu.sessions._invocation_lifecycle import (
    released_invocation_evidence as released_invocation_evidence,
)
from cayu.sessions._invocation_lifecycle import (
    replay_invocation_lifecycle_command_from_state as replay_invocation_lifecycle_command_from_state,
)
from cayu.sessions._invocation_lifecycle import (
    require_invocation_admission_source_authority as require_invocation_admission_source_authority,
)
from cayu.sessions._invocation_lifecycle import (
    require_invocation_command_authority as require_invocation_command_authority,
)
from cayu.sessions._invocation_lifecycle import (
    require_invocation_lifecycle_release_capacity as require_invocation_lifecycle_release_capacity,
)
from cayu.sessions._invocation_lifecycle import (
    require_invocation_rebind_lineage as require_invocation_rebind_lineage,
)
from cayu.sessions._invocation_lifecycle import (
    require_invocation_release_store_authority as require_invocation_release_store_authority,
)
from cayu.sessions._invocation_lifecycle import (
    require_released_invocation_command_authority as require_released_invocation_command_authority,
)
from cayu.sessions._invocation_lifecycle import (
    superseding_invocation_admission_digest_from_state as superseding_invocation_admission_digest_from_state,
)
from cayu.sessions.base import (
    RuntimePublicationMutation,
    SessionInvocationAdmission,
    SessionRunFenced,
    SessionStore,
    _current_session_run_epoch,
    _deactivate_session_interaction,
    _deactivate_session_run_fence,
    runtime_publication_checkpoint_mutation,
)
from cayu.sessions.checkpoints import (
    ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
    CHECKPOINT_SCHEMA_VERSION_KEY,
    INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
)
from cayu.sessions.records import Session, SessionStatus, copy_session
from cayu.tools.discovery import (
    initial_tool_discovery_operation_records_from_initialization,
)
from cayu.tools.exposure import (
    ToolCapabilityCeiling,
    tool_capability_ceiling_from_session_metadata,
)
from cayu.tools.grants import PreparedTargetedToolGrant

_INVOCATION_CONTEXT_AUTHORITY_TOKEN = object()


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
        "_recovery_claim_id",
        "_registered_agent",
        "_registered_environment",
        "_registered_provider",
        "_request_loop_policies",
        "_runtime_hooks",
        "_targeted_tool_grants",
        "_tool_capability_ceiling",
        "_validated_profile",
        "_work_attempt",
    )

    _active_profile: ActiveInvocationExecutionProfile
    _authority_token: object
    _binding: InvocationBinding
    _budget_policy: BudgetPolicy | None
    _loop_policies: tuple[LoopPolicy, ...]
    _registered_agent: runtime_records.RegisteredAgentState
    _registered_environment: runtime_records.RegisteredEnvironment | None
    _registered_provider: runtime_records.RegisteredProvider
    _recovery_claim_id: str | None
    _request_loop_policies: tuple[LoopPolicy, ...]
    _runtime_hooks: tuple[runtime_records.RegisteredRuntimeHook, ...]
    _targeted_tool_grants: tuple[PreparedTargetedToolGrant, ...]
    _tool_capability_ceiling: ToolCapabilityCeiling
    _validated_profile: ExecutionProfileIdentity
    _work_attempt: WorkAttemptInvocationAuthority | None

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
        _recovery_claim_id: str | None = None,
        _validated_profile: ExecutionProfileIdentity | None = None,
        _authority_token: object = None,
        _work_attempt: WorkAttemptInvocationAuthority | None = None,
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
        object.__setattr__(self, "_recovery_claim_id", _recovery_claim_id)
        object.__setattr__(self, "_validated_profile", _validated_profile)
        object.__setattr__(self, "_authority_token", _authority_token)
        object.__setattr__(self, "_work_attempt", _work_attempt)
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
        if self.recovery_claim_id is not None:
            require_durable_clean_nonblank(self.recovery_claim_id, "recovery_claim_id")
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
        if self.work_attempt is not None:
            if type(self.work_attempt) is not WorkAttemptInvocationAuthority:
                raise TypeError("Invocation context requires authenticated work-attempt authority.")
            admission = self.work_attempt.admission
            if (
                admission.session_id != binding.session_id
                or admission.session_invocation.session_instance_id != binding.session_instance_id
                or admission.interaction_id != binding.interaction_id
                or admission.source_execution_profile_fingerprint != self.profile.fingerprint
            ):
                raise ValueError("Work-attempt authority conflicts with its invocation binding.")
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

    def require_runtime_authority(self) -> None:
        """Revalidate the runtime seal without granting authority to copied fields."""
        self._validate()

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

    @property
    def recovery_claim_id(self) -> str | None:
        """Return runtime-authenticated terminal-recovery ownership, if any."""

        return self._recovery_claim_id

    @property
    def work_attempt(self) -> WorkAttemptInvocationAuthority | None:
        """Return the runtime-only task owner, never caller-supplied metadata."""
        return self._work_attempt

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
                or session.runtime_build_provenance != binding.runtime_build_provenance
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
            or session.runtime_build_provenance != binding.runtime_build_provenance
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
                runtime_build_provenance=binding.runtime_build_provenance,
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
            recovery_claim_id=self.recovery_claim_id,
            work_attempt=self.work_attempt,
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
                recovery_claim_id=self.recovery_claim_id,
                work_attempt=self.work_attempt,
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
            recovery_claim_id=self.recovery_claim_id,
            work_attempt=self.work_attempt,
        )

    @staticmethod
    def _require_monotonic_environment_resolution(
        current: runtime_records.RegisteredEnvironment,
        replacement: runtime_records.RegisteredEnvironment,
    ) -> None:
        """Accept only binding or owned-result settlement, never handle substitution."""

        if current.execution_candidate is None and replacement.execution_candidate is not None:
            if (
                replacement.factory is not current.factory
                or replacement.environment is not current.environment
                or replacement.bound_workspace is not current.bound_workspace
                or replacement.binding_payload is not current.binding_payload
                or replacement.unclaimed_factory_result is not current.unclaimed_factory_result
                or replacement.retained_factory_result is not current.retained_factory_result
                or replacement.preserve_factory_allocation != current.preserve_factory_allocation
                or replacement.live_allocation_fingerprint != current.live_allocation_fingerprint
                or replacement.binding_generation_id != current.binding_generation_id
                or replacement.workspace_mutation_fence is not current.workspace_mutation_fence
                or replacement.environment_exposure is not None
            ):
                raise ValueError("Invocation context received an invalid environment selection.")
            return

        if (
            replacement.factory is not None
            or replacement.binding_generation_id != current.binding_generation_id
            or replacement.workspace_mutation_fence is not current.workspace_mutation_fence
            or replacement.execution_candidate != current.execution_candidate
            or replacement.execution_candidate_declared != current.execution_candidate_declared
            or replacement.execution_environment_authority
            is not current.execution_environment_authority
            or replacement.live_allocation_fingerprint != current.live_allocation_fingerprint
            or (
                current.environment_exposure is not None
                and replacement.environment_exposure is not current.environment_exposure
            )
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
            or session.runtime_build_provenance != binding.runtime_build_provenance
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
                runtime_build_provenance=session.runtime_build_provenance,
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
            recovery_claim_id=self.recovery_claim_id,
            work_attempt=self.work_attempt,
        )

    def with_queued_interaction(
        self,
        session: Session,
        *,
        active_profile: ActiveInvocationExecutionProfile,
    ) -> InvocationContext:
        """Transfer this exact same-epoch authority to an atomically started turn."""

        if type(session) is not Session:
            raise TypeError("session must be a Session.")
        if type(active_profile) is not ActiveInvocationExecutionProfile:
            raise TypeError("active_profile must be an ActiveInvocationExecutionProfile.")
        binding = self.binding
        if not isinstance(binding, AdmittedInvocationBinding):
            raise ValueError("Queued interaction handoff requires admitted invocation authority.")
        if (
            session.id != binding.session_id
            or session.instance_id != binding.session_instance_id
            or session.run_epoch != binding.run_epoch
            or session.agent_name != binding.agent_name
            or session.provider_name != binding.provider_name
            or session.model != binding.model
            or session.runtime_name != binding.runtime_name
            or session.runtime_version != binding.runtime_version
            or session.runtime_build_provenance != binding.runtime_build_provenance
            or session.environment_name != binding.environment_name
            or active_profile.session_id != session.id
            or active_profile.run_epoch != session.run_epoch
            or active_profile.profile != self.profile
            or active_profile.interaction_id == binding.interaction_id
        ):
            raise ValueError("Queued interaction handoff conflicts with invocation authority.")
        rebound_active_profile = active_profile.model_copy(
            update={"profile": self.profile},
            deep=False,
        )
        return _authenticated_invocation_context(
            active_profile=rebound_active_profile,
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
                runtime_build_provenance=session.runtime_build_provenance,
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
            recovery_claim_id=self.recovery_claim_id,
            work_attempt=self.work_attempt,
        )

    def without_recovery_claim(self) -> InvocationContext:
        """Drop a settled recovery claim while preserving invocation authority."""

        if self.recovery_claim_id is None:
            return self
        return _authenticated_invocation_context(
            active_profile=self.active_profile,
            binding=self.binding,
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
            work_attempt=self.work_attempt,
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
    recovery_claim_id: str | None = None,
    work_attempt: WorkAttemptInvocationAuthority | None = None,
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
        _recovery_claim_id=recovery_claim_id,
        _validated_profile=validated_profile,
        _authority_token=_INVOCATION_CONTEXT_AUTHORITY_TOKEN,
        _work_attempt=work_attempt,
    )


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
    from cayu.runtime._external_wait_receipts import external_execution_origin

    receipt = _invocation_lifecycle_command_receipt(
        command,
        active_profile=active_profile,
        result_session=result_session,
        external_execution_origin=external_execution_origin(command, ledger),
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
    from cayu.sessions._session_continuation_store import pending_admission_receipt_identities

    pending_admissions = pending_admission_receipt_identities(result_session, updated)
    compacted_receipts = _compact_invocation_lifecycle_receipts(
        retained,
        retained_command_identity=receipt.command_identity,
        release_capacity_command_identity=release_capacity_command_identity,
        result_session=result_session,
        pending_admission_identities=pending_admissions,
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
            pending_admission_identities=pending_admissions,
        )
        next_ledger = _InvocationLifecycleReceiptLedger(
            receipts=compacted_receipts,
            release_capacity_command_identity=release_capacity_command_identity,
        )
    updated[INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY] = next_ledger.model_dump(mode="json")
    return updated


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


def retire_released_invocation_context(evidence: InvocationReleaseEvidence) -> None:
    """Retire stale local context after the caller authenticates exact release.

    This internal phase handoff accepts engine-validated release evidence or
    its exact retained settlement receipt, never public caller-created proof.
    It changes no durable ownership and must not clear a newer invocation's
    epoch or another session's context.
    """
    epoch = _current_session_run_epoch(evidence.session_id)
    if epoch is not None and epoch <= evidence.run_epoch:
        _deactivate_session_interaction(evidence.session_id)
        _deactivate_session_run_fence(evidence.session_id)


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
    return replay_invocation_lifecycle_command_from_state(session, checkpoint, replay_command)


def invocation_checkpoint_state_sha256(
    checkpoint: dict[str, Any] | None,
) -> str:
    # Continuation ownership has its own atomic index and receipt. Its event
    # sequence may advance immediately before admission; it is not invocation
    # lifecycle source authority and must not make an otherwise exact command
    # stale.
    if checkpoint is not None:
        checkpoint = copy_durable_json_object(checkpoint, "invocation lifecycle checkpoint")
        from cayu.collaboration._session_export_store import ROOT_KEY as EXPORT_ROOT_KEY
        from cayu.sessions._producer_checkpoint import ROOT_KEY as PRODUCER_ROOT_KEY
        from cayu.sessions._session_continuation_store import ROOT_KEY

        checkpoint.pop(ROOT_KEY, None)
        # Export ownership is projected separately by native stores and is not
        # visible to lifecycle mutation callbacks. Including it only in the
        # preparation read makes an unchanged exported session falsely stale.
        # The export owner still validates/preserves its root independently.
        checkpoint.pop(EXPORT_ROOT_KEY, None)
        # Producer ownership has its own exact native CAS transition. Ordinary
        # runtime reads hide this root; it cannot change their lifecycle digest.
        checkpoint.pop(PRODUCER_ROOT_KEY, None)
    return sha256(
        canonical_durable_json_bytes(
            checkpoint,
            "invocation lifecycle source checkpoint",
        )
    ).hexdigest()


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
    temporary_service = None
    if type(copied) is AdmitInvocationCommand:
        from cayu.sessions._temporary_continuation_scope import require_temporary_admission

        temporary_service = require_temporary_admission(copied)
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

        def admit_checkpoint(session: Session, checkpoint: dict[str, Any] | None, now: datetime):
            from cayu.runtime._producer_output_store import require_native_admission
            from cayu.sessions._session_continuation_store import require_admission_claim

            require_native_admission(checkpoint, copied, now=now)
            require_admission_claim(session, checkpoint, copied)
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
            updated = _checkpoint_after_predecessor_terminal_decision(
                session,
                checkpoint,
                expected_active_profile=copied.expected_active_profile,
            )
            return _apply_checkpoint_patch(copied.checkpoint_patch, updated)

        def record_admit_result(session: Session, checkpoint: dict[str, Any] | None):
            return checkpoint_with_invocation_lifecycle_receipt(
                checkpoint,
                copied,
                active_profile=copied.target_active_profile,
                result_session=session,
            )

        from cayu.sessions._browser_control_checkpoint import browser_control_checkpoint_read_scope

        try:
            # Admission hashes the complete source checkpoint, just like rebind.
            # Authorize this internal callback's private read without granting
            # generic transforms visibility or browser-control mutation authority.
            with (
                _invocation_lifecycle_authority_mutation_scope(),
                browser_control_checkpoint_read_scope(copied.session_id),
            ):
                session = await store.admit_session_invocation(
                    copied.session_id,
                    admission=SessionInvocationAdmission(
                        from_statuses=frozenset(copied.expected_statuses),
                        checkpoint_transform=None,
                        store_time_checkpoint_transform=admit_checkpoint,
                        result_checkpoint_transform=record_admit_result,
                        execution_profile=copied.target_active_profile.profile,
                        interaction_source_messages=copied.interaction_source_messages,
                        tool_capability_ceiling=copied.tool_capability_ceiling,
                        interaction_started_event=copied.interaction_started_event,
                        continued_interaction_id=copied.continued_interaction_id,
                        defer_interaction_source=copied.defer_interaction_source,
                        model_transition=copied.model_transition,
                        execution_profile_decision=copied.execution_profile_decision,
                        adopted_runtime_identity=copied.adopted_runtime_identity,
                        expected_active_invocation_profile=copied.expected_active_profile,
                        allow_pending_initial_interaction=(
                            copied.allow_pending_initial_interaction
                        ),
                        temporary_service_admission=temporary_service,
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
            from cayu.sessions._producer_checkpoint import ROOT_KEY, NativeProducerIndex

            if checkpoint is not None and ROOT_KEY in checkpoint:
                producer = NativeProducerIndex.model_validate(checkpoint[ROOT_KEY])
                if producer.cleanup_receipt is not None:
                    from cayu.runtime._producer_output_store import require_successor_invocation

                    require_successor_invocation(producer, copied.target_active_profile)
                elif producer.paused_stop is not None:
                    raise SessionRunFenced("The exact paused producer was durably stopped.")
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

        from cayu.sessions._browser_control_checkpoint import browser_control_checkpoint_read_scope

        try:
            # This typed command compares the complete source checkpoint. Give
            # its internal transform the same private view used in preparation;
            # generic callbacks still cannot read or mutate browser authority.
            with (
                _invocation_lifecycle_authority_mutation_scope(),
                browser_control_checkpoint_read_scope(copied.session_id),
            ):
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


def _rebound_active_invocation_profile(
    session: Session,
    snapshot: ActiveInvocationExecutionProfile,
) -> ActiveInvocationExecutionProfile:
    """Carry one validated profile into the run epoch claimed for recovery."""

    if snapshot.session_id != session.id:
        raise RuntimeError("Recovery profile authority belongs to a different session.")
    return snapshot.model_copy(update={"run_epoch": session.run_epoch})


def reconstruct_invocation_context(
    *,
    runtime_hooks: tuple[runtime_records.RegisteredRuntimeHook, ...],
    loop_policies: tuple[LoopPolicy, ...],
    session: Session,
    execution_profile_snapshot: ActiveInvocationExecutionProfile,
    registered_agent: runtime_records.RegisteredAgentState,
    registered_provider: runtime_records.RegisteredProvider,
    registered_environment: runtime_records.RegisteredEnvironment | None,
    budget_policy: BudgetPolicy | None,
    request_loop_policies: tuple[LoopPolicy, ...] = (),
    recovery_claim_id: str | None = None,
    work_attempt: WorkAttemptInvocationAuthority | None = None,
) -> InvocationContext:
    """Authenticate restart-resolved collaborators before recovery effects."""

    active_profile = _rebound_active_invocation_profile(
        session,
        execution_profile_snapshot,
    )
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
            runtime_build_provenance=session.runtime_build_provenance,
            environment_name=session.environment_name,
        ),
        validated_profile=active_profile.profile,
        registered_agent=registered_agent,
        registered_provider=registered_provider,
        registered_environment=registered_environment,
        runtime_hooks=runtime_hooks,
        loop_policies=loop_policies,
        request_loop_policies=request_loop_policies,
        budget_policy=budget_policy,
        tool_capability_ceiling=tool_capability_ceiling_from_session_metadata(session.metadata),
        recovery_claim_id=recovery_claim_id,
        work_attempt=work_attempt,
    )
