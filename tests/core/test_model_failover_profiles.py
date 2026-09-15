from __future__ import annotations

from dataclasses import replace

import pytest

from cayu import ModelFailoverPolicy, ModelTarget
from cayu.evals.testing import ScriptedModelProvider
from cayu.providers.operations import ProviderOperationMode
from cayu.runtime._execution_profile_admission import (
    bind_model_failover_execution_profile,
    reconstruct_model_failover_execution_profile,
    resolve_model_failover_execution_profile,
)
from cayu.runtime._runtime_records import RegisteredProvider
from cayu.runtime.build_provenance import current_runtime_build_provenance
from cayu.runtime.execution_profiles import (
    ActiveInvocationExecutionProfile,
    ExecutionProfileComponentClass,
    ExecutionProfileIdentity,
    ExecutionProfileIdentityStrength,
    build_execution_profile_identity,
    changed_execution_profile_components,
    execution_profile_with_component,
    execution_profile_with_tool_capability_ceiling,
    inherited_execution_profile_component_changes,
)
from cayu.sessions._model_failover import ModelFailoverCandidate, ModelFailoverPlan
from cayu.vaults.redaction import SecretRedactor


def _profile(provider_name, model, *, runtime_version="test", **changes):
    return build_execution_profile_identity(
        runtime_name="cayu",
        runtime_version=runtime_version,
        runtime_build_provenance=current_runtime_build_provenance(),
        provider_name=provider_name,
        model=model,
        durable_system_prompt="answer safely",
        direct_tools=(),
        tool_catalogue_revision="sha256:" + "0" * 64,
        **changes,
    )


def _plan(primary, backup):
    return ModelFailoverPlan(
        candidates=(
            ModelFailoverCandidate(
                provider_name="primary",
                model="small",
                execution_profile_fingerprint=primary.fingerprint,
                execution_mode="synchronous",
            ),
            ModelFailoverCandidate(
                provider_name="backup",
                model="large",
                execution_profile_fingerprint=backup.fingerprint,
                execution_mode="synchronous",
            ),
        ),
        max_total_attempts=5,
    )


def test_failover_plan_changes_request_policy_not_invocation_authority():
    primary = _profile("primary", "small")
    backup = _profile("backup", "large")
    plan = _plan(primary, backup)
    bound = bind_model_failover_execution_profile(plan=plan, candidate_profiles=(primary, backup))
    assert changed_execution_profile_components(primary, bound) == (
        ExecutionProfileComponentClass.PROVIDER_REQUEST_POLICY,
    )
    assert bound != primary
    assert bound.schema_version == 7 and primary.schema_version == 6
    assert bound.model_failover is not None and bound.model_failover.plan == plan
    assert bound.runtime_build_provenance == primary.runtime_build_provenance
    assert bound.egress_authority == primary.egress_authority
    assert (
        bind_model_failover_execution_profile(
            plan=ModelFailoverPlan.model_validate_json(plan.model_dump_json()),
            candidate_profiles=(primary, backup),
        )
        == bound
    )


def test_inherited_ceiling_rebinds_complete_plan_without_other_component_changes():
    primary, backup = _profile("primary", "small"), _profile("backup", "large")
    original = bind_model_failover_execution_profile(
        plan=_plan(primary, backup), candidate_profiles=(primary, backup)
    )
    changed = execution_profile_with_tool_capability_ceiling(original, ("one",))
    assert inherited_execution_profile_component_changes(original, changed) == (
        ExecutionProfileComponentClass.TOOL_VIEW_GRANTS,
    )
    assert set(changed_execution_profile_components(original, changed)) == {
        ExecutionProfileComponentClass.TOOL_VIEW_GRANTS,
        ExecutionProfileComponentClass.PROVIDER_REQUEST_POLICY,
    }
    assert changed.model_failover is not None and original.model_failover is not None
    assert changed.model_failover.plan != original.model_failover.plan
    assert changed.model_failover.strength == original.model_failover.strength
    assert execution_profile_with_tool_capability_ceiling(changed, ("one",)) == changed
    reconstructed = ExecutionProfileIdentity.model_validate_json(changed.model_dump_json())
    assert reconstructed == changed
    assert inherited_execution_profile_component_changes(original, reconstructed) == (
        ExecutionProfileComponentClass.TOOL_VIEW_GRANTS,
    )


@pytest.mark.parametrize(
    ("change", "component"),
    [
        (
            {"provider_adapter": {"version": "changed"}},
            ExecutionProfileComponentClass.PROVIDER_ADAPTER,
        ),
        (
            {"provider_request_policy": {"options": {"new": True}}},
            ExecutionProfileComponentClass.PROVIDER_REQUEST_POLICY,
        ),
        ({"structured_output": {"new": True}}, ExecutionProfileComponentClass.STRUCTURED_OUTPUT),
        ({"execution_policies": {"new": True}}, ExecutionProfileComponentClass.EXECUTION_POLICIES),
    ],
)
def test_inherited_comparison_exposes_backup_only_component_changes(change, component):
    primary, backup = _profile("primary", "small"), _profile("backup", "large")
    original = bind_model_failover_execution_profile(
        plan=_plan(primary, backup), candidate_profiles=(primary, backup)
    )
    replacement = _profile("backup", "large", **change)
    changed = bind_model_failover_execution_profile(
        plan=_plan(primary, replacement), candidate_profiles=(primary, replacement)
    )
    assert inherited_execution_profile_component_changes(original, changed) == (component,)


@pytest.mark.parametrize("change", ["cap", "mode", "order"])
def test_inherited_comparison_never_hides_plan_control_changes(change):
    primary, backup = _profile("primary", "small"), _profile("backup", "large")
    plan = _plan(primary, backup)
    original = bind_model_failover_execution_profile(
        plan=plan, candidate_profiles=(primary, backup)
    )
    payload = plan.payload()
    profiles = (primary, backup)
    if change == "cap":
        payload["max_total_attempts"] += 1
    elif change == "mode":
        for candidate in payload["candidates"]:
            candidate["execution_mode"] = "background"
    else:
        payload["candidates"].reverse()
        profiles = (backup, primary)
    changed = bind_model_failover_execution_profile(
        plan=ModelFailoverPlan.model_validate(payload), candidate_profiles=profiles
    )
    assert ExecutionProfileComponentClass.PROVIDER_REQUEST_POLICY in (
        inherited_execution_profile_component_changes(original, changed)
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"provider_adapter": {"version": 2}},
        {"provider_request_policy": {"options": {"output": "different"}}},
        {"structured_output": {"schema": {"type": "object"}}},
        {"execution_policies": {"targeted_projection": "different"}},
    ],
)
def test_candidate_drift_is_bound_and_cannot_reuse_original_plan(changes):
    primary = _profile("primary", "small")
    backup = _profile("backup", "large")
    changed = _profile("backup", "large", **changes)
    plan = _plan(primary, backup)
    original = bind_model_failover_execution_profile(
        plan=plan, candidate_profiles=(primary, backup)
    )
    with pytest.raises(ValueError, match="conflicts"):
        bind_model_failover_execution_profile(plan=plan, candidate_profiles=(primary, changed))
    revised = bind_model_failover_execution_profile(
        plan=_plan(primary, changed), candidate_profiles=(primary, changed)
    )
    assert revised != original


def test_plan_cannot_bind_a_profile_for_another_provider_or_model():
    primary = _profile("primary", "small")
    wrong_backup = _profile("different-provider", "large")
    plan = _plan(primary, wrong_backup)
    with pytest.raises(ValueError, match="conflicts"):
        bind_model_failover_execution_profile(plan=plan, candidate_profiles=(primary, wrong_backup))


@pytest.mark.parametrize(
    ("options", "strength"),
    [
        ({"provider_adapter_process_local": True}, ExecutionProfileIdentityStrength.PROCESS_LOCAL),
        (
            {"provider_request_policy_process_local": True},
            ExecutionProfileIdentityStrength.PROCESS_LOCAL,
        ),
        (
            {"provider_adapter_application_versioned": True},
            ExecutionProfileIdentityStrength.APPLICATION_VERSIONED,
        ),
    ],
)
def test_candidate_identity_strength_is_never_upgraded_by_binding(options, strength):
    primary = _profile("primary", "small")
    backup = _profile("backup", "large", **options)
    bound = bind_model_failover_execution_profile(
        plan=_plan(primary, backup), candidate_profiles=(primary, backup)
    )
    assert (
        bound.component(ExecutionProfileComponentClass.PROVIDER_REQUEST_POLICY).strength is strength
    )


def test_bound_profile_is_not_accepted_as_its_own_unbound_candidate():
    primary = _profile("primary", "small")
    backup = _profile("backup", "large")
    plan = _plan(primary, backup)
    bound = bind_model_failover_execution_profile(plan=plan, candidate_profiles=(primary, backup))
    with pytest.raises(ValueError, match="conflicts"):
        bind_model_failover_execution_profile(plan=plan, candidate_profiles=(bound, backup))


def test_resolution_freezes_registered_candidates_before_extension_profile_callbacks():
    primary = RegisteredProvider("primary", ScriptedModelProvider([], name="primary"))
    backup = RegisteredProvider("backup", ScriptedModelProvider([], name="backup"))
    replacement = RegisteredProvider("backup", ScriptedModelProvider([], name="backup"))
    registry = {"backup": backup}
    observations = []
    profiles = []

    def resolve_profile(registered, model):
        observations.append(registered)
        registry["backup"] = replacement
        profile = _profile(registered.name, model)
        profiles.append(profile)
        return profile

    resolution = resolve_model_failover_execution_profile(
        policy=ModelFailoverPolicy(fallbacks=(ModelTarget(provider_name="backup", model="large"),)),
        primary=ModelTarget(provider_name="primary", model="small"),
        registered_primary=primary,
        resolve_provider=registry.__getitem__,
        resolve_candidate_profile=resolve_profile,
        redactor=SecretRedactor(),
    )
    assert observations == [primary, backup]
    assert resolution.registered_providers == (primary, backup)
    assert resolution.registered_providers[1] is backup
    assert resolution.candidate_profiles == tuple(profiles)
    assert resolution.candidate_profiles[1] is not profiles[1]
    assert resolution.candidate_profiles[1].components[0] is not profiles[1].components[0]
    original = resolution.profile
    object.__setattr__(profiles[1], "fingerprint", "c" * 64)
    assert resolution.profile == original
    assert isinstance(primary.provider, ScriptedModelProvider)
    assert isinstance(backup.provider, ScriptedModelProvider)
    assert not primary.provider.requests and not backup.provider.requests


@pytest.mark.parametrize("during_callback", [False, True])
def test_resolution_rejects_changed_execution_mode_even_with_unchanged_profile(during_callback):
    class ModeProvider(ScriptedModelProvider):
        mode = ProviderOperationMode.SYNCHRONOUS

        @property
        def provider_operation_mode(self):
            return self.mode

    primary_adapter = ModeProvider([], name="primary")
    backup_adapter = ModeProvider([], name="backup")
    primary = RegisteredProvider("primary", primary_adapter)
    backup = RegisteredProvider("backup", backup_adapter)
    registry = {"primary": primary, "backup": backup}

    def resolve_profile(registered, model):
        if during_callback:
            backup_adapter.mode = ProviderOperationMode.BACKGROUND
        return _profile(registered.name, model)

    def resolve():
        return resolve_model_failover_execution_profile(
            policy=ModelFailoverPolicy(
                fallbacks=(ModelTarget(provider_name="backup", model="large"),)
            ),
            primary=ModelTarget(provider_name="primary", model="small"),
            registered_primary=primary,
            resolve_provider=registry.__getitem__,
            resolve_candidate_profile=resolve_profile,
            redactor=SecretRedactor(),
        )

    if during_callback:
        with pytest.raises(ValueError, match="mode changed"):
            resolve()
        return
    resolution = resolve()
    # Both providers changing together must not evade the original snapshot by
    # continuing to match one another. Their application profile IDs stay equal.
    primary_adapter.mode = backup_adapter.mode = ProviderOperationMode.BACKGROUND
    with pytest.raises(ValueError, match="mode changed"):
        replace(resolution)
    changed = resolve()
    assert changed.candidate_profiles == resolution.candidate_profiles
    assert changed.profile.fingerprint != resolution.profile.fingerprint
    with pytest.raises(ValueError, match="admitted profile"):
        reconstruct_model_failover_execution_profile(
            expected_profile=resolution.profile,
            resolve_provider=registry.__getitem__,
            resolve_candidate_profile=resolve_profile,
            redactor=SecretRedactor(),
        )


def test_admitted_profile_reconstructs_plan_before_first_stage_exists():
    primary = _profile("primary", "small")
    backup = _profile("backup", "large")
    bound = bind_model_failover_execution_profile(
        plan=_plan(primary, backup), candidate_profiles=(primary, backup)
    )
    active = ActiveInvocationExecutionProfile(
        session_id="session", interaction_id="interaction", run_epoch=1, profile=bound
    )
    reopened = ActiveInvocationExecutionProfile.model_validate_json(active.model_dump_json())
    assert reopened == active
    assert reopened.profile.model_failover is not None
    assert reopened.profile.model_failover.plan == _plan(primary, backup)
    assert tuple(
        item.as_profile() for item in reopened.profile.model_failover.candidate_profiles
    ) == (primary, backup)
    assert reopened.profile.model_failover is not bound.model_failover
    assert (
        execution_profile_with_component(
            reopened.profile,
            reopened.profile.component(ExecutionProfileComponentClass.FINALIZATION),
        )
        == bound
    )
    altered = _profile("primary", "small", finalization={"different": True})
    with pytest.raises(ValueError, match="primary"):
        execution_profile_with_component(
            bound, altered.component(ExecutionProfileComponentClass.FINALIZATION)
        )


@pytest.mark.parametrize(
    "alteration",
    [
        "missing",
        "schema",
        "cap",
        "target",
        "profile",
        "order",
        "primary_policy",
        "strength",
        "candidate_missing",
        "candidate_order",
        "candidate_component",
        "candidate_nested",
    ],
)
def test_bound_profile_rejects_reconstruction_material_conflicts(alteration):
    primary = _profile("primary", "small")
    backup = _profile("backup", "large")
    bound = bind_model_failover_execution_profile(
        plan=_plan(primary, backup), candidate_profiles=(primary, backup)
    )
    data = bound.model_dump(mode="json")
    binding = data["model_failover"]
    if alteration == "missing":
        del data["model_failover"]
    elif alteration == "schema":
        data["schema_version"] = 6
    elif alteration == "cap":
        binding["plan"]["max_total_attempts"] += 1
    elif alteration == "target":
        binding["plan"]["candidates"][1]["model"] = "different"
    elif alteration == "profile":
        binding["plan"]["candidates"][1]["execution_profile_fingerprint"] = "c" * 64
    elif alteration == "order":
        binding["plan"]["candidates"].reverse()
    elif alteration == "primary_policy":
        binding["primary_request_policy"]["fingerprint"] = "c" * 64
    elif alteration == "strength":
        binding["strength"] = "process_local"
    elif alteration == "candidate_missing":
        del binding["candidate_profiles"]
    elif alteration == "candidate_order":
        binding["candidate_profiles"].reverse()
    elif alteration == "candidate_component":
        binding["candidate_profiles"][1]["components"][0]["fingerprint"] = "c" * 64
    elif alteration == "candidate_nested":
        binding["candidate_profiles"][1]["model_failover"] = {}
    with pytest.raises(ValueError):
        ExecutionProfileIdentity.model_validate(data)


@pytest.mark.parametrize("drift", [False, True])
def test_reconstruction_uses_recorded_targets_and_detects_fallback_only_drift(drift):
    primary = _profile("primary", "small")
    backup = _profile("backup", "large")
    expected = bind_model_failover_execution_profile(
        plan=_plan(primary, backup), candidate_profiles=(primary, backup)
    )
    reopened = ExecutionProfileIdentity.model_validate_json(expected.model_dump_json())
    registry = {
        name: RegisteredProvider(name, ScriptedModelProvider([], name=name))
        for name in ("primary", "backup", "new-default")
    }
    lookups = []

    def resolve_provider(name):
        lookups.append(name)
        return registry[name]

    def resolve_profile(registered, model):
        if drift and registered.name == "backup":
            return _profile(registered.name, model, provider_adapter={"version": "changed"})
        return _profile(registered.name, model)

    def reconstruct():
        return reconstruct_model_failover_execution_profile(
            expected_profile=reopened,
            resolve_provider=resolve_provider,
            resolve_candidate_profile=resolve_profile,
            redactor=SecretRedactor(),
        )

    if drift:
        with pytest.raises(ValueError, match="conflicts"):
            reconstruct()
    else:
        resolution = reconstruct()
        assert resolution.profile == expected
        assert resolution.plan == _plan(primary, backup)
    assert lookups == ["primary", "backup"]


@pytest.mark.parametrize("invalid", ["duplicate", "secret", "registration", "profile"])
def test_resolution_rejects_invalid_candidate_configuration_without_provider_dispatch(invalid):
    primary = RegisteredProvider("primary", ScriptedModelProvider([], name="primary"))
    backup = RegisteredProvider("backup", ScriptedModelProvider([], name="backup"))
    target = ModelTarget(provider_name="backup", model="large")
    if invalid == "duplicate":
        target = ModelTarget(provider_name="primary", model="small")
    if invalid == "secret":
        target = ModelTarget(provider_name="backup", model="credential-canary")
    callbacks = []

    def resolve_provider(name):
        callbacks.append("registration")
        return primary if invalid == "registration" else backup

    def resolve_profile(registered, model):
        callbacks.append("profile")
        return _profile("wrong-provider" if invalid == "profile" else registered.name, model)

    with pytest.raises(ValueError):
        resolve_model_failover_execution_profile(
            policy=ModelFailoverPolicy(fallbacks=(target,)),
            primary=ModelTarget(provider_name="primary", model="small"),
            registered_primary=primary,
            resolve_provider=resolve_provider,
            resolve_candidate_profile=resolve_profile,
            redactor=SecretRedactor("credential-canary"),
        )
    if invalid in {"duplicate", "secret"}:
        assert not callbacks
    elif invalid == "registration":
        assert callbacks == ["registration"]
    assert isinstance(primary.provider, ScriptedModelProvider)
    assert isinstance(backup.provider, ScriptedModelProvider)
    assert not primary.provider.requests and not backup.provider.requests
