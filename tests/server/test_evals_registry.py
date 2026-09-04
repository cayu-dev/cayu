from __future__ import annotations

import asyncio
from decimal import Decimal
from pathlib import Path

import pytest
from tests.evals.test_corpus_execution import (
    _model_judge_target,
    _price_book,
    _provider,
    _target,
)

import cayu.server.evals_registry as evals_registry_module
from cayu import (
    AgentSpec,
    CayuApp,
    CorpusExecutionLimits,
    CorpusTarget,
    EvalRunCostBudget,
    EvalRunInvocation,
    EvalScenarioRunInvocation,
    Message,
    RunDefaults,
    RunRequest,
    SecretRedactor,
    default_price_book,
)
from cayu.evals.execution import evaluation_target_identity
from cayu.evals.execution_profiles import EvalExecutionProfilePolicyV1
from cayu.project_control_plane import ProjectEvalJudgeConfiguration
from cayu.runtime.config import DEFAULT_MAX_STEPS, CayuConfig
from cayu.runtime.invocation import (
    InvocationOrigin,
    InvocationOriginClaim,
    InvocationOriginTrust,
    SessionExecutionSource,
)
from cayu.runtime.stop_policy import RunLimits
from cayu.server.evals_registry import (
    DEFAULT_EVAL_PROFILE_ID,
    EvalTargetRegistry,
    derive_eval_target_key,
    explicit_eval_target_registry,
    generated_eval_target_registry,
    target_for_eval_invocation,
)


def _generated_registry(*, release_id: str = "release-one") -> EvalTargetRegistry:
    target = _target(_provider())
    app = target.app
    app.register_agent(AgentSpec(name="beta", model="fixture-model"))
    manifest = app.describe()
    registry = generated_eval_target_registry(
        app,
        project_id="registry-project",
        application_release_id=release_id,
        app_manifest_fingerprint=manifest.fingerprint,
    )
    assert registry is not None
    return registry


def _copy_target(
    target: CorpusTarget,
    *,
    request_base=None,
    bootstrap_messages=None,
    limits: CorpusExecutionLimits | None = None,
) -> CorpusTarget:
    return CorpusTarget(
        key=target.key,
        app=target.app,
        request_base=target.request_base if request_base is None else request_base,
        bootstrap_messages=(
            target.bootstrap_messages if bootstrap_messages is None else bootstrap_messages
        ),
        application_release_id=target.application_release_id,
        evidence_policy=target.evidence_policy,
        price_book=target.price_book,
        model_judges=target.model_judges,
        limits=target.limits if limits is None else limits,
    )


def _prepared_profile(target: CorpusTarget):
    registry = explicit_eval_target_registry(target)
    return asyncio.run(registry.prepare_execution_profile(target.key))


def test_generated_target_keys_are_stable_unambiguous_and_release_independent() -> None:
    baseline = derive_eval_target_key(
        project_id="project",
        agent_name="agent",
        profile_id="default",
    )

    assert baseline == derive_eval_target_key(
        project_id="project",
        agent_name="agent",
        profile_id="default",
    )
    assert baseline == "eval.b1cbe3c1b491a3bd35c3c259200cca26d82c526bb6ebde8a248ccb834af80dbc"
    assert len(baseline) == 69
    assert (
        len(
            {
                baseline,
                derive_eval_target_key(
                    project_id="project-two",
                    agent_name="agent",
                    profile_id="default",
                ),
                derive_eval_target_key(
                    project_id="project",
                    agent_name="agent-two",
                    profile_id="default",
                ),
                derive_eval_target_key(
                    project_id="project",
                    agent_name="agent",
                    profile_id="isolated",
                ),
                derive_eval_target_key(
                    project_id="ab",
                    agent_name="c",
                    profile_id="default",
                ),
                derive_eval_target_key(
                    project_id="a",
                    agent_name="bc",
                    profile_id="default",
                ),
            }
        )
        == 6
    )


def test_generated_registry_maps_each_agent_to_normal_authority_without_serializing_app() -> None:
    first = _generated_registry(release_id="release-one")
    second = _generated_registry(release_id="release-two")

    assert first.target_keys == second.target_keys
    assert len(first.target_keys) == 2
    catalog = first.catalog()
    assert [entry.agent_name for entry in catalog.items] == ["agent", "beta"]
    assert {entry.profile_id for entry in catalog.items} == {DEFAULT_EVAL_PROFILE_ID}
    assert {entry.project_id for entry in catalog.items} == {"registry-project"}
    assert {entry.application_release_id for entry in catalog.items} == {"release-one"}
    assert catalog.default_target_key == first.target_keys[0]
    assert "CayuApp" not in catalog.model_dump_json()
    assert all(entry.judge_profiles == () for entry in catalog.items)
    assert repr(first) == "EvalTargetRegistry(target_count=2)"

    for entry in catalog.items:
        assert entry.max_trials == 1
        assert entry.max_concurrency == 1
        assert entry.max_timeout_seconds == 3_600
        assert entry.max_steps == DEFAULT_MAX_STEPS
        assert entry.cost_budget_available is False
        assert entry.cost_budget_currencies == ()
        assert entry.judge_profiles == ()
        assert entry.judge_profile_routes == ()
        runtime_target = first.get(entry.target_key)
        assert runtime_target is not None
        assert runtime_target.request_base.agent_name == entry.agent_name
        assert runtime_target.request_base.messages == []
        assert runtime_target.application_release_id == "release-one"


def test_eval_registry_resolves_app_max_steps_and_preserves_explicit_override() -> None:
    async def scenario() -> None:
        app = CayuApp(config=CayuConfig(run=RunDefaults(max_steps=128)), enable_logging=False)
        app.register_provider(_provider(), default=True)
        app.register_agent(AgentSpec(name="agent", model="fixture-model"))
        manifest = app.describe()
        generated = generated_eval_target_registry(
            app,
            project_id="configured-registry-project",
            application_release_id="configured-registry-release",
            app_manifest_fingerprint=manifest.fingerprint,
        )
        assert generated is not None

        generated_entry = generated.catalog().items[0]
        generated_profile = await generated.prepare_execution_profile(generated_entry.target_key)
        assert generated_entry.max_steps == 128
        assert generated_profile.snapshot.ceilings.max_steps == 128

        explicit_target = CorpusTarget(
            key="configured-explicit-target",
            app=app,
            request_base=RunRequest(
                agent_name="agent",
                messages=[],
                max_steps=80,
            ),
            application_release_id="configured-registry-release",
            limits=CorpusExecutionLimits(),
        )
        explicit = explicit_eval_target_registry(explicit_target)
        explicit_entry = explicit.catalog().items[0]
        explicit_profile = await explicit.prepare_execution_profile(explicit_target.key)
        assert explicit_entry.max_steps == 80
        assert explicit_profile.snapshot.ceilings.max_steps == 80

    asyncio.run(scenario())


def test_generated_registry_publishes_only_an_explicit_declarative_judge() -> None:
    target = _target(_provider())
    manifest = target.app.describe()
    configuration = ProjectEvalJudgeConfiguration(
        provider_name=target.app.list_providers()[0],
        model="fixture-model",
        privacy_policy="public-and-transcript",
        allow_same_model=True,
        timeout_seconds=45,
        max_input_tokens=4096,
        max_output_tokens=1024,
        max_total_tokens=5120,
        max_estimated_cost="0.1",
        cost_currency="USD",
    )
    pricing = _price_book()

    registry = generated_eval_target_registry(
        target.app,
        project_id="registry-project",
        application_release_id="release-one",
        app_manifest_fingerprint=manifest.fingerprint,
        price_book=pricing,
        judge_configuration=configuration,
    )

    assert registry is not None
    entry = registry.catalog().items[0]
    assert len(entry.judge_profiles) == 1
    profile = entry.judge_profiles[0]
    assert profile.key == "project-default-judge"
    assert profile.label == "Project default judge"
    assert profile.provider_name == configuration.provider_name
    assert profile.model == "fixture-model"
    assert profile.allowed_evidence == (
        "final_output",
        "transcript",
        "public_reference",
    )
    assert profile.timeout_seconds == 45
    assert profile.max_input_tokens == 4096
    assert profile.max_output_tokens == 1024
    assert profile.max_total_tokens == 5120
    assert profile.max_estimated_cost == "0.1"
    assert profile.cost_currency == "USD"
    assert profile.pricing_profile_fingerprint is not None
    assert profile.same_model_use == "allowed_and_labeled"
    assert len(entry.judge_profile_routes) == 1
    assert entry.judge_profile_routes[0].candidate_route_relation == "same_model"
    runtime_target = registry.get(entry.target_key)
    assert runtime_target is not None
    assert len(runtime_target.model_judges) == 1
    judge = runtime_target.model_judges[0]
    assert judge.app is not target.app
    assert judge.app.list_providers() == (configuration.provider_name,)
    assert not judge.app.get_agent(judge.agent_name).tools
    assert judge.price_book == pricing


def test_generated_registry_rejects_a_priced_judge_without_project_pricing() -> None:
    target = _target(_provider())
    manifest = target.app.describe()

    with pytest.raises(ValueError, match="cost ceiling requires project Evals pricing"):
        generated_eval_target_registry(
            target.app,
            project_id="registry-project",
            application_release_id="release-one",
            app_manifest_fingerprint=manifest.fingerprint,
            judge_configuration=ProjectEvalJudgeConfiguration(
                provider_name=target.app.list_providers()[0],
                model="fixture-model",
                privacy_policy="public-only",
                allow_same_model=False,
                max_estimated_cost="0.1",
                cost_currency="USD",
            ),
        )


def test_generated_registry_rejects_secret_bearing_or_unregistered_judge_routes() -> None:
    secret = "private-judge-route"
    app = CayuApp(enable_logging=False, secret_redactor=SecretRedactor(secret))
    app.register_provider(_provider(), default=True)
    app.register_agent(AgentSpec(name="agent", model="fixture-model"))
    manifest = app.describe()

    with pytest.raises(ValueError, match="contains a workload secret") as secret_error:
        generated_eval_target_registry(
            app,
            project_id="registry-project",
            application_release_id="release-one",
            app_manifest_fingerprint=manifest.fingerprint,
            judge_configuration=ProjectEvalJudgeConfiguration(
                provider_name=app.list_providers()[0],
                model=secret,
                privacy_policy="public-only",
                allow_same_model=False,
            ),
        )
    assert secret not in str(secret_error.value)

    with pytest.raises(KeyError, match="Provider not registered"):
        generated_eval_target_registry(
            app,
            project_id="registry-project",
            application_release_id="release-one",
            app_manifest_fingerprint=manifest.fingerprint,
            judge_configuration=ProjectEvalJudgeConfiguration(
                provider_name="unregistered",
                model="fixture-model",
                privacy_policy="public-only",
                allow_same_model=False,
            ),
        )


def test_explicit_catalog_publishes_safe_exact_judge_profiles() -> None:
    judge, _ = _model_judge_target()
    target = _target(_provider(), model_judges=(judge,))

    entry = explicit_eval_target_registry(target).catalog().items[0]

    assert len(entry.judge_profiles) == 1
    profile = entry.judge_profiles[0]
    assert profile.key == "quality-judge"
    assert profile.provider_name == "scripted"
    assert profile.model == "judge-model"
    assert profile.allowed_evidence == ("final_output", "public_reference")
    assert profile.same_model_use == "forbidden"
    assert len(entry.judge_profile_routes) == 1
    route = entry.judge_profile_routes[0]
    assert (route.judge_profile_key, route.judge_profile_revision) == (
        profile.key,
        profile.revision,
    )
    assert route.candidate_route_relation == "independent_model"
    serialized = entry.model_dump_json()
    assert "provider_options" not in serialized
    assert "system_prompt" not in serialized
    assert "api_key" not in serialized


def test_resolved_catalog_publishes_deterministic_safe_execution_profiles() -> None:
    registry = _generated_registry()

    first = asyncio.run(registry.resolved_catalog())
    second = asyncio.run(registry.resolved_catalog())

    assert tuple(item.execution_profile_ready for item in first.items) == (True, True)
    assert tuple(item.execution_profile_diagnostics for item in first.items) == ((), ())
    assert tuple(item.execution_profile for item in first.items) == tuple(
        item.execution_profile for item in second.items
    )
    for item in first.items:
        profile = item.execution_profile
        assert profile is not None
        assert profile.target_key == item.target_key
        assert profile.candidate.agent_name == item.agent_name
        assert profile.candidate.provider_name == "scripted"
        assert profile.candidate.model == "fixture-model"
        assert len(profile.candidate.runtime_execution_profile_fingerprint) == 64
        assert profile.fixture_strategy == "none"
        assert profile.reset_strategy == "fresh_session_only"
        assert profile.effect_posture == "ordinary_application_authority"
        assert profile.isolation_revision is None
        runtime_target = registry.get(item.target_key)
        assert runtime_target is not None
        assert profile.evidence_policy == runtime_target.evidence_policy
        assert profile.target_material.kind == "structural_sha256"
        assert profile.target_material.process_scope is None
        assert profile.ceilings.max_cases == runtime_target.limits.max_cases
        assert profile.ceilings.max_trials == 1
        assert profile.ceilings.max_concurrency == 1
        assert (
            profile.ceilings.max_bootstrap_messages == runtime_target.limits.max_bootstrap_messages
        )
        assert profile.ceilings.max_total_input_chars == runtime_target.limits.max_total_input_chars
        assert (
            profile.ceilings.max_compiled_input_chars
            == runtime_target.limits.max_compiled_input_chars
        )
        assert profile.revision.startswith("sha256:")
        changed = profile.model_dump(mode="json")
        changed_candidate = changed["candidate"]
        assert isinstance(changed_candidate, dict)
        changed_candidate["model"] = "different-model"
        with pytest.raises(ValueError, match="revision does not match"):
            type(profile).model_validate(changed)


def test_execution_profile_binds_complete_target_inputs_and_limits() -> None:
    target = _target(_provider())
    baseline = _prepared_profile(target)
    changed_request = target.request_base.model_copy(
        update={
            "labels": {"eval-mode": "changed"},
            "metadata": {"candidate-variant": "changed"},
        }
    )
    variants = (
        _copy_target(
            target,
            bootstrap_messages=(Message.text("system", "A different candidate bootstrap."),),
        ),
        _copy_target(target, request_base=changed_request),
        *(
            _copy_target(
                target,
                limits=CorpusExecutionLimits.model_validate(
                    {**target.limits.model_dump(mode="python"), field_name: value}
                ),
            )
            for field_name, value in (
                ("max_cases", 1),
                ("max_bootstrap_messages", 1),
                ("max_total_input_chars", 128),
                ("max_compiled_input_chars", 128),
            )
        ),
    )

    for variant in variants:
        prepared = _prepared_profile(variant)
        assert prepared.snapshot.revision != baseline.snapshot.revision
        assert prepared.binding != baseline.binding

    serialized = baseline.snapshot.model_dump_json()
    assert "Follow the refund policy." not in serialized
    assert '"max_cases":1000' in serialized


def test_private_target_material_uses_process_local_hmac_without_leaking_values() -> None:
    secret = "private-bootstrap-token"
    target = _target(_provider(), secret_redactor=SecretRedactor(secret))
    first = _copy_target(
        target,
        bootstrap_messages=(Message.text("system", f"Use {secret} for fixture A."),),
    )
    second = _copy_target(
        target,
        bootstrap_messages=(Message.text("system", f"Use {secret} for fixture B."),),
    )

    first_profile = _prepared_profile(first)
    replayed_profile = _prepared_profile(first)
    second_profile = _prepared_profile(second)
    restarted_base = _target(_provider(), secret_redactor=SecretRedactor(secret))
    restarted = _copy_target(
        restarted_base,
        bootstrap_messages=(Message.text("system", f"Use {secret} for fixture A."),),
    )
    restarted_profile = _prepared_profile(restarted)

    assert first_profile.snapshot.target_material.kind == "process_local_hmac_sha256"
    assert first_profile.snapshot.target_material.process_scope is not None
    assert first_profile.snapshot == replayed_profile.snapshot
    assert first_profile.snapshot.revision != second_profile.snapshot.revision
    assert (
        first_profile.snapshot.target_material.process_scope
        != restarted_profile.snapshot.target_material.process_scope
    )
    assert (
        first_profile.snapshot.target_material.fingerprint
        != restarted_profile.snapshot.target_material.fingerprint
    )
    assert secret not in first_profile.snapshot.model_dump_json()


def test_resolved_catalog_reports_application_identity_drift_without_fallback() -> None:
    registry = _generated_registry()
    target = registry.get(registry.default_target_key)
    assert target is not None
    target.app.register_agent(AgentSpec(name="late-agent", model="fixture-model"))

    catalog = asyncio.run(registry.resolved_catalog())

    assert all(not item.execution_profile_ready for item in catalog.items)
    assert all(item.execution_profile is None for item in catalog.items)
    assert {
        diagnostic.code
        for item in catalog.items
        for diagnostic in item.execution_profile_diagnostics
    } == {"application_identity_changed"}


def test_profile_preparation_rejects_application_drift_during_resolution(monkeypatch) -> None:
    target = _target(_provider())
    registry = explicit_eval_target_registry(target)
    original = evals_registry_module.prepare_eval_execution_profile

    async def mutate_after_preparation(*args, **kwargs):
        prepared = await original(*args, **kwargs)
        target.app.register_agent(AgentSpec(name="late-agent", model="fixture-model"))
        return prepared

    monkeypatch.setattr(
        evals_registry_module,
        "prepare_eval_execution_profile",
        mutate_after_preparation,
    )

    catalog = asyncio.run(registry.resolved_catalog())

    assert all(not item.execution_profile_ready for item in catalog.items)
    assert {
        diagnostic.code
        for item in catalog.items
        for diagnostic in item.execution_profile_diagnostics
    } == {"application_identity_changed"}


def test_repeated_execution_requires_an_application_isolation_contract() -> None:
    target = _target(_provider())

    with pytest.raises(ValueError, match="application-managed reset"):
        EvalExecutionProfilePolicyV1(max_trials=2)

    policy = EvalExecutionProfilePolicyV1(
        fixture_strategy="application_managed",
        reset_strategy="application_managed",
        effect_posture="isolated_application_authority",
        isolation_revision="sha256:" + "a" * 64,
        max_trials=2,
        max_concurrency=2,
    )
    registry = explicit_eval_target_registry(target, policy=policy)
    profile = asyncio.run(registry.resolved_catalog()).items[0].execution_profile

    assert registry.get(target.key) is target
    assert profile is not None
    assert profile.ceilings.max_trials == 2
    assert profile.ceilings.max_concurrency == 2
    assert profile.isolation_revision == policy.isolation_revision


def test_generated_registry_exposes_only_server_owned_pricing_availability() -> None:
    target = _target(_provider())
    pricing = _price_book()
    manifest = target.app.describe()

    registry = generated_eval_target_registry(
        target.app,
        project_id="registry-project",
        application_release_id="release-one",
        app_manifest_fingerprint=manifest.fingerprint,
        price_book=pricing,
    )

    assert registry is not None
    assert {entry.cost_budget_available for entry in registry.catalog().items} == {True}
    assert {entry.cost_budget_currencies for entry in registry.catalog().items} == {("USD",)}
    assert all(
        target is not None and target.price_book == pricing
        for target in (registry.get(key) for key in registry.target_keys)
    )


def test_generated_registry_does_not_advertise_unpriced_target_models() -> None:
    target = _target(_provider())
    manifest = target.app.describe()

    registry = generated_eval_target_registry(
        target.app,
        project_id="registry-project",
        application_release_id="release-one",
        app_manifest_fingerprint=manifest.fingerprint,
        price_book=default_price_book(),
    )

    assert registry is not None
    assert {entry.cost_budget_available for entry in registry.catalog().items} == {False}
    assert {entry.cost_budget_currencies for entry in registry.catalog().items} == {()}
    runtime_target = registry.get(registry.default_target_key)
    assert runtime_target is not None
    with pytest.raises(ValueError, match="target model"):
        target_for_eval_invocation(
            runtime_target,
            EvalRunInvocation(cost_budget=EvalRunCostBudget(max_estimated_cost=Decimal("0.25"))),
        )


def test_explicit_registry_rejects_execution_policy_beyond_target_authority() -> None:
    target = _target(_provider())
    target = target.model_copy(
        update={"limits": target.limits.model_copy(update={"max_trials": 1, "max_concurrency": 1})}
    )
    policy = EvalExecutionProfilePolicyV1(
        reset_strategy="application_managed",
        isolation_revision="sha256:" + "1" * 64,
        max_trials=2,
    )

    with pytest.raises(ValueError, match="exceeds its runtime target authority"):
        explicit_eval_target_registry(target, policy=policy)


def test_generated_registry_does_not_crash_when_priced_target_has_no_provider() -> None:
    app = CayuApp()
    app.register_agent(AgentSpec(name="agent", model="gpt-5"))
    manifest = app.describe()

    registry = generated_eval_target_registry(
        app,
        project_id="registry-project",
        application_release_id="release-one",
        app_manifest_fingerprint=manifest.fingerprint,
        price_book=default_price_book(),
    )

    assert registry is not None
    entry = registry.catalog().items[0]
    assert entry.cost_budget_available is False
    assert entry.cost_budget_currencies == ()
    resolved = asyncio.run(registry.resolved_catalog()).items[0]
    assert resolved.execution_profile_ready is False
    assert resolved.execution_profile is None
    assert [item.code for item in resolved.execution_profile_diagnostics] == [
        "runtime_authority_unavailable"
    ]
    assert resolved.execution_profile_diagnostics[0].message == (
        "The current runtime execution profile is unavailable."
    )
    runtime_target = registry.get(entry.target_key)
    assert runtime_target is not None
    with pytest.raises(ValueError, match="target model"):
        target_for_eval_invocation(
            runtime_target,
            EvalRunInvocation(cost_budget=EvalRunCostBudget(max_estimated_cost=Decimal("0.25"))),
        )


def test_eval_invocation_can_only_contract_target_authority() -> None:
    target = _target(_provider(), price_book=_price_book())
    target = target.model_copy(
        update={
            "request_base": target.request_base.model_copy(
                update={"limits": RunLimits(max_total_tokens=50, scope="session")}
            )
        }
    )
    invocation = EvalRunInvocation(
        source=SessionExecutionSource.HTTP_RUN,
        origin=InvocationOrigin(
            trust=InvocationOriginTrust.SERVER_VERIFIED,
            subject="eval-operator",
            tenant="tenant-one",
        ),
        max_steps=3,
        limits=RunLimits(max_total_tokens=100, max_tool_calls=2, scope="run"),
        cost_budget=EvalRunCostBudget(max_estimated_cost=Decimal("0.25"), currency="usd"),
    )

    effective = target_for_eval_invocation(target, invocation)

    assert effective.request_base.max_steps == target.request_base.max_steps == 1
    assert effective.request_base.limits.max_total_tokens == 50
    assert effective.request_base.limits.max_tool_calls == 2
    assert effective.request_base._runtime_invocation_source is SessionExecutionSource.HTTP_RUN
    assert effective.request_base._verified_invocation_origin == invocation.origin
    assert len(effective.request_base.budget_limits) == 1
    budget = effective.request_base.budget_limits[0]
    assert invocation.cost_budget is not None
    assert budget.max_estimated_cost == invocation.cost_budget.max_estimated_cost
    assert budget.currency == "USD"
    assert budget.allow_unpriced is False
    assert effective.app is target.app
    assert target.request_base.budget_limits == ()

    unsupported_currency = invocation.model_copy(
        update={
            "cost_budget": EvalRunCostBudget(
                max_estimated_cost=Decimal("0.25"),
                currency="EUR",
            )
        }
    )
    with pytest.raises(ValueError, match="requested currency"):
        target_for_eval_invocation(target, unsupported_currency)

    host_asserted_target = target.model_copy(
        update={
            "request_base": target.request_base.model_copy(
                update={"invocation_origin": InvocationOriginClaim(subject="sdk-host")}
            )
        }
    )
    with pytest.raises(ValueError, match="cannot inherit a host-asserted SDK origin"):
        target_for_eval_invocation(host_asserted_target, invocation)


def test_execution_profile_preserves_valid_session_scoped_target_limits() -> None:
    target = _target(_provider())
    target = target.model_copy(
        update={
            "request_base": target.request_base.model_copy(
                update={"limits": RunLimits(max_total_tokens=50, scope="session")}
            )
        }
    )
    registry = explicit_eval_target_registry(target)

    profile = asyncio.run(registry.prepare_execution_profile(target.key)).snapshot

    assert profile.ceilings.run_limits.max_total_tokens == 50
    assert profile.ceilings.run_limits.scope == "session"


def test_eval_scenario_invocation_selects_only_compatible_published_environment() -> None:
    target = _target(_provider())
    assert target.request_base.environment_name is None
    invocation = EvalRunInvocation(
        scenario=EvalScenarioRunInvocation(
            scenario_revision="sha256:" + "1" * 64,
            binding_revision="sha256:" + "2" * 64,
            environment_name="eval-environment",
            trials=1,
            timeout_seconds=30,
        )
    )

    effective = target_for_eval_invocation(target, invocation)

    assert effective.request_base.environment_name == "eval-environment"
    pinned = target.model_copy(
        update={
            "request_base": target.request_base.model_copy(
                update={"environment_name": "production"}
            )
        }
    )
    with pytest.raises(ValueError, match="environment conflicts"):
        target_for_eval_invocation(pinned, invocation)


def test_generated_registry_preserves_project_root_manifest_provenance() -> None:
    target = _target(_provider())
    project_root = Path(__file__).resolve().parents[2]
    rooted_manifest = target.app.describe(project_root=project_root)
    unrooted_manifest = target.app.describe()

    assert rooted_manifest.fingerprint != unrooted_manifest.fingerprint
    with pytest.raises(ValueError, match="catalog manifest"):
        generated_eval_target_registry(
            target.app,
            project_id="registry-project",
            application_release_id="release-one",
            app_manifest_fingerprint=unrooted_manifest.fingerprint,
            app_manifest_project_root=project_root,
        )

    registry = generated_eval_target_registry(
        target.app,
        project_id="registry-project",
        application_release_id="release-one",
        app_manifest_fingerprint=rooted_manifest.fingerprint,
        app_manifest_project_root=project_root,
    )
    assert registry is not None
    entry = registry.catalog().items[0]
    registration = registry.registration(entry.target_key)
    assert registration is not None
    runtime_identity = evaluation_target_identity(
        registration.target,
        project_root=registration.manifest_project_root,
    )

    assert entry.app_manifest_fingerprint == rooted_manifest.fingerprint
    assert runtime_identity.app_manifest == rooted_manifest


def test_generated_registry_rejects_hash_collisions(monkeypatch) -> None:
    monkeypatch.setattr(
        "cayu.server.evals_registry.derive_eval_target_key",
        lambda **_kwargs: "eval." + "0" * 64,
    )

    with pytest.raises(ValueError, match="key collision"):
        _generated_registry()


def test_explicit_v1_target_is_adapted_without_changing_its_authority() -> None:
    target = _target(_provider())
    registry = explicit_eval_target_registry(target)
    entry = registry.catalog().items[0]

    assert registry.target_keys == (target.key,)
    assert registry.get(target.key) is target
    assert entry.target_key == target.key
    assert entry.agent_name == target.request_base.agent_name
    assert entry.profile_id == "explicit"
    assert entry.project_id is None
    assert entry.source == "explicit"
