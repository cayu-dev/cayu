from __future__ import annotations

import asyncio

from tests.evals.test_corpus_execution import _corpus, _provider

from cayu import AgentSpec, CayuApp, RunRequest, Tool, ToolEffect, ToolResult, ToolSpec
from cayu.evals import CorpusTarget
from cayu.evals.corpus import (
    CorpusUserMessageSpec,
    EvaluationEvidencePolicySpec,
    RootStatusAssertionSpec,
    RunInputSpec,
)
from cayu.evals.execution import run_corpus_suite
from cayu.evals.execution_comparison import eval_result_compatibility
from cayu.evals.results import EvalResultProjectionV2, eval_result_projection
from cayu.evals.suite_authoring import (
    EvalCaseDraftV1,
    EvalSimpleInputStimulusV1,
    EvalSuiteDraftV1,
    compile_eval_suite_draft,
    eval_suite_selection,
)
from cayu.evals.suite_preflight import (
    EvalCandidateLaunchExposure,
    compile_authored_suite_run_exposure,
)
from cayu.runtime import ExecutionProfileBehaviorIdentity
from cayu.server.evals_registry import explicit_eval_target_registry


class Lookup(Tool):
    def __init__(self, *, declared=False, name="lookup"):
        self.spec = ToolSpec(
            name=name,
            description="Lookup",
            effect=ToolEffect.NONE,
            input_schema={"type": "object"},
            max_terminal_payload_bytes=1024,
            execution_profile_identity=(
                ExecutionProfileBehaviorIdentity(
                    name="lookup", behavior_version="1", implementation_version="1"
                )
                if declared
                else None
            ),
        )
        super().__init__()

    async def run(self, ctx, args):
        return ToolResult(content="ok")


def prepared(*, declared=False, target=None, tool_count=1):
    if target is None:
        app = CayuApp(enable_logging=False)
        app.register_provider(_provider(trials=1), default=True)
        app.register_agent(
            AgentSpec(name="agent", model="fixture-model"),
            tools=[
                Lookup(declared=declared, name="lookup" if tool_count == 1 else f"t{i:04d}")
                for i in range(tool_count)
            ],
        )
        target = CorpusTarget(
            key="refund-agent",
            app=app,
            request_base=RunRequest(agent_name="agent", messages=[], max_steps=1),
            application_release_id="release",
            evidence_policy=EvaluationEvidencePolicySpec.standard(),
        )
    profile = asyncio.run(
        explicit_eval_target_registry(target).prepare_execution_profile(target.key)
    ).snapshot
    return target, profile


def exposure(profile):
    document = compile_eval_suite_draft(
        EvalSuiteDraftV1(
            id="identity",
            name="Identity",
            target_key=profile.target_key,
            cases=(
                EvalCaseDraftV1(
                    id="case",
                    name="Case",
                    stimulus=EvalSimpleInputStimulusV1(
                        input=RunInputSpec(messages=(CorpusUserMessageSpec(text="Evaluate"),))
                    ),
                    assertions=(RootStatusAssertionSpec(id="completed", expected="completed"),),
                ),
            ),
        )
    )
    return compile_authored_suite_run_exposure(
        document,
        eval_suite_selection(document),
        (EvalCandidateLaunchExposure(case_ids=("case",), execution_profile=profile),),
        judge_profiles=(),
        candidate_pricing_profile_fingerprint=None,
    )


def projection_with_profile(projection, profile):
    accepted = exposure(profile)
    return EvalResultProjectionV2.model_validate(
        {
            **projection.model_dump(mode="json"),
            "accepted_exposure_revision": accepted.revision,
            "accepted_exposure_comparison_revision": accepted.comparison_revision,
            "process_local_identities_truncated": accepted.process_local_identities_truncated,
            "process_local_identities": [
                identity.model_dump(mode="json") for identity in accepted.process_local_identities
            ],
        }
    )


def test_runtime_profile_diagnostics_survive_restart_and_exposure_projection():
    target, first = prepared()
    _, same = prepared(target=target)
    _, restarted = prepared()
    assert [item.component for item in first.process_local_identities] == ["tool:agent/lookup"]
    assert first.process_local_identities == same.process_local_identities
    assert first.process_local_identities != restarted.process_local_identities
    fresh = asyncio.run(run_corpus_suite(target, _corpus(trials=1), "refund-regressions"))
    projection = eval_result_projection(fresh)
    baseline = projection_with_profile(projection, first)
    assert eval_result_compatibility(baseline, projection_with_profile(projection, same)).comparable
    current = projection_with_profile(projection, restarted)
    restored = EvalResultProjectionV2.model_validate_json(current.model_dump_json())
    comparison = eval_result_compatibility(baseline, restored)
    assert not comparison.comparable
    assert "accepted_exposure_contract_mismatch" in comparison.reasons
    assert "process_local_execution_identity" in comparison.reasons
    assert comparison.process_local_components == ("tool:agent/lookup",)


def test_declared_profiles_are_portable_and_diagnostics_do_not_change_semantics():
    _, first = prepared(declared=True)
    _, second = prepared(declared=True)
    assert first.process_local_identities == second.process_local_identities == ()
    assert first.comparison_revision == second.comparison_revision
    assert exposure(first).comparison_revision == exposure(second).comparison_revision
    _, undeclared = prepared()
    stripped = type(undeclared).create(
        **{
            **{
                name: getattr(undeclared, name)
                for name in type(undeclared).model_fields
                if name not in {"schema_version", "revision"}
            },
            "process_local_identities": (),
            "process_local_identities_truncated": True,
        }
    )
    assert stripped.revision != undeclared.revision
    assert stripped.comparison_revision == undeclared.comparison_revision
    assert exposure(stripped).comparison_revision == exposure(undeclared).comparison_revision


def test_same_scope_unrelated_drift_is_not_called_a_process_identity_change():
    target, profile = prepared()
    fresh = asyncio.run(run_corpus_suite(target, _corpus(trials=1), "refund-regressions"))
    baseline = projection_with_profile(eval_result_projection(fresh), profile)
    current = baseline.model_copy(
        update={"accepted_exposure_comparison_revision": "sha256:" + "f" * 64}
    )
    result = eval_result_compatibility(baseline, current)
    assert result.reasons == ("accepted_exposure_contract_mismatch",)
    assert result.process_local_components == ()
    absent = current.model_copy(update={"process_local_identities": ()})
    assert eval_result_compatibility(baseline, absent).process_local_components == ()


def test_serialized_manifest_cannot_invent_booted_component_evidence():
    from cayu.runtime.checks import check_manifest
    from cayu.runtime.manifest import AppManifest

    target, _ = prepared()
    manifest = target.app.describe()
    assert any(
        item.code == "EVAL_EXECUTION_IDENTITY_PROCESS_LOCAL"
        for item in check_manifest(manifest).diagnostics
    )
    restored = AppManifest.model_validate_json(manifest.model_dump_json())
    assert not any(
        item.code == "EVAL_EXECUTION_IDENTITY_PROCESS_LOCAL"
        for item in check_manifest(restored).diagnostics
    )


def test_real_process_restart_preserves_diagnostic_evidence():
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    import cayu

    command = "from tests.evals.test_process_local_comparison import prepared; print(prepared()[1].model_dump_json())"
    environment = {
        **os.environ,
        "PYTHONPATH": f"{Path(cayu.__file__).parents[1]}{os.pathsep}{Path.cwd()}",
    }
    profiles = [
        json.loads(
            subprocess.run(
                [sys.executable, "-c", command],
                env=environment,
                check=True,
                capture_output=True,
                text=True,
                timeout=45,
            ).stdout
        )
        for _ in range(2)
    ]
    assert profiles[0]["process_local_identities"][0]["component"] == "tool:agent/lookup"
    assert profiles[0]["process_local_identities"] != profiles[1]["process_local_identities"]
    assert (
        profiles[0]["candidate"]["runtime_execution_profile_fingerprint"]
        != profiles[1]["candidate"]["runtime_execution_profile_fingerprint"]
    )


def test_runtime_provider_diagnostics_follow_exact_owned_classification():
    from cayu.evals.testing import ScriptedModelProvider
    from cayu.vaults import SecretRedactor

    class CustomProvider(ScriptedModelProvider):
        pass

    def profile(provider, *, redactor=None):
        app = CayuApp(enable_logging=False, secret_redactor=redactor)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="agent", model="fixture-model"))
        target = CorpusTarget(
            key="provider",
            app=app,
            request_base=RunRequest(agent_name="agent", messages=[]),
            application_release_id="release",
            evidence_policy=EvaluationEvidencePolicySpec.standard(),
        )
        return prepared(target=target)[1]

    assert profile(ScriptedModelProvider([])).process_local_identities == ()
    custom = profile(CustomProvider([], name="custom"))
    assert [item.component for item in custom.process_local_identities] == ["provider:custom"]
    # A maintained provider's private configuration is process bound for a
    # different reason; it must not produce advice to declare missing behavior.
    from cayu.providers import ModelStreamEvent

    private = profile(
        ScriptedModelProvider(
            [
                (
                    ModelStreamEvent.text_delta("secret-canary"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                )
            ]
        ),
        redactor=SecretRedactor("secret-canary"),
    )
    assert private.process_local_identities == ()
    assert "secret-canary" not in private.model_dump_json()


def test_bound_failover_provider_identity_is_named_even_before_dispatch():
    from cayu.evals.testing import ScriptedModelProvider
    from cayu.sessions import ModelFailoverPolicy, ModelTarget

    class CustomProvider(ScriptedModelProvider):
        pass

    app = CayuApp(enable_logging=False)
    app.register_provider(ScriptedModelProvider([], name="primary"), default=True)
    app.register_provider(CustomProvider([], name="backup"))
    app.register_agent(AgentSpec(name="agent", model="fixture-model"))
    target = CorpusTarget(
        key="failover",
        app=app,
        application_release_id="release",
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
        request_base=RunRequest(
            agent_name="agent",
            messages=[],
            failover=ModelFailoverPolicy(
                fallbacks=(ModelTarget(provider_name="backup", model="fixture-model"),),
            ),
        ),
    )
    profile = prepared(target=target)[1]
    assert [item.component for item in profile.process_local_identities] == ["provider:backup"]


def test_large_tool_catalogue_preserves_registry_readiness():
    _, profile = prepared(tool_count=1000)
    assert len(profile.model_dump_json().encode()) < 16_384
    assert profile.process_local_identities_truncated
    assert 0 < len(profile.process_local_identities) <= 8


def test_large_tool_catalogue_preserves_complete_benchmark_invocation(tmp_path):
    from cayu.evals.benchmark_campaign import prepare_benchmark_campaign
    from cayu.evals.benchmark_package import BenchmarkPackageV1, LoadedBenchmarkPackage
    from cayu.evals.runner import EvalPlan
    from cayu.evals.suite_authoring import (
        EvalCaseDraftV2,
        EvalSuiteDraftV3,
        compile_eval_suite_draft_v3,
    )

    target, _ = prepared(tool_count=300)
    suite = compile_eval_suite_draft_v3(
        EvalSuiteDraftV3(
            id="large",
            name="Large",
            target_key=target.key,
            cases=(
                EvalCaseDraftV2(
                    id="case",
                    name="Case",
                    stimulus=EvalSimpleInputStimulusV1(
                        input=RunInputSpec(messages=(CorpusUserMessageSpec(text="Evaluate"),))
                    ),
                    assertions=(RootStatusAssertionSpec(id="completed", expected="completed"),),
                ),
            ),
        )
    )
    package = BenchmarkPackageV1.create(
        id="large",
        version="1",
        scorer_id="root",
        scorer_version="1",
        suite=suite,
    )
    campaign = asyncio.run(
        prepare_benchmark_campaign(
            LoadedBenchmarkPackage(package=package, root=tmp_path, file_contents=()),
            EvalPlan(corpus_target=target),
        )
    )
    for run in campaign.campaign.runs:
        invocation = run.spec.invocation
        assert invocation.execution_profile_snapshot.process_local_identities_truncated
        assert invocation.authored_suite_exposure.process_local_identities_truncated
        assert len(invocation.model_dump_json().encode()) < 16_384


def test_identity_preview_count_bytes_and_complete_scope_sets():
    import pytest

    from cayu._validation import canonical_durable_json_bytes
    from cayu.execution_profiles import (
        ProcessLocalExecutionIdentity,
        bounded_process_local_identities,
        validate_process_local_identity_preview,
    )

    def identity(name, scope="a"):
        return ProcessLocalExecutionIdentity(component=f"tool:{name}", process_scope=scope * 64)

    for count in (0, 7, 8, 9):
        evidence = tuple(identity(f"t{i}") for i in range(count))
        preview, truncated = bounded_process_local_identities(evidence)
        assert len(preview) == min(count, 8)
        assert truncated is (count > 8)
        assert validate_process_local_identity_preview(preview) == preview
    # Exercise the actual UTF-8 bound, not a character-count approximation.
    base = (identity("a"), identity("b"))
    overhead = (
        len(canonical_durable_json_bytes([i.model_dump(mode="json") for i in base], "test")) - 2
    )
    for delta in (-1, 0, 1):
        name_bytes = 2048 + delta - overhead
        evidence = (identity("a" * 900), identity("b" * (name_bytes - 900)))
        encoded = canonical_durable_json_bytes(
            [i.model_dump(mode="json") for i in evidence], "test"
        )
        assert len(encoded) == 2048 + delta
        preview, truncated = bounded_process_local_identities(evidence)
        assert truncated is (delta > 0)
        assert (
            len(canonical_durable_json_bytes([i.model_dump(mode="json") for i in preview], "test"))
            <= 2048
        )
    unicode_evidence = (identity("é" * 900), identity("界" * 900))
    preview, truncated = bounded_process_local_identities(unicode_evidence)
    assert truncated
    assert len(preview) == 1
    with pytest.raises(ValueError, match="bounded"):
        validate_process_local_identity_preview(unicode_evidence)
    # An over-budget multi-scope component is omitted as a whole; retaining
    # only one scope could falsely imply disjoint owners on comparison.
    evidence = (*(identity("same", str(i)) for i in range(9)), identity("z"))
    preview, truncated = bounded_process_local_identities(evidence)
    assert truncated and preview == (identity("z"),)


def test_exposure_preview_does_not_infer_missing_scopes_from_truncated_profiles():
    from cayu.execution_profiles import ProcessLocalExecutionIdentity

    _, profile = prepared()
    document = compile_eval_suite_draft(
        EvalSuiteDraftV1(
            id="multi",
            name="Multi",
            target_key=profile.target_key,
            cases=tuple(
                EvalCaseDraftV1(
                    id=case,
                    name=case,
                    stimulus=EvalSimpleInputStimulusV1(
                        input=RunInputSpec(messages=(CorpusUserMessageSpec(text="Evaluate"),))
                    ),
                    assertions=(RootStatusAssertionSpec(id="completed", expected="completed"),),
                )
                for case in ("a", "b")
            ),
        )
    )

    def with_names(names):
        return type(profile).create(
            **{
                **{
                    name: getattr(profile, name)
                    for name in type(profile).model_fields
                    if name not in {"schema_version", "revision"}
                },
                "process_local_identities": tuple(
                    ProcessLocalExecutionIdentity(
                        component=name,
                        process_scope="b" * 64,
                    )
                    for name in names
                ),
                "process_local_identities_truncated": True,
            }
        )

    for names, expected in (
        (["tool:agent/other"], ("tool:agent/other",)),
        (["tool:agent/lookup"], ("tool:agent/lookup",) * 2),
    ):
        accepted = compile_authored_suite_run_exposure(
            document,
            eval_suite_selection(document),
            (
                EvalCandidateLaunchExposure(case_ids=("a",), execution_profile=profile),
                EvalCandidateLaunchExposure(case_ids=("b",), execution_profile=with_names(names)),
            ),
            judge_profiles=(),
            candidate_pricing_profile_fingerprint=None,
        )
        assert accepted.process_local_identities_truncated
        assert tuple(i.component for i in accepted.process_local_identities) == expected
        assert len(accepted.model_dump_json().encode()) < 4096


def test_overlapping_recorded_scopes_never_claim_a_restart_cause():
    from cayu.execution_profiles import ProcessLocalExecutionIdentity

    target, profile = prepared()
    fresh = asyncio.run(run_corpus_suite(target, _corpus(trials=1), "refund-regressions"))
    baseline = projection_with_profile(eval_result_projection(fresh), profile)

    def evidence(scopes):
        return tuple(
            ProcessLocalExecutionIdentity(
                component="tool:agent/lookup",
                process_scope=scope * 64,
            )
            for scope in scopes
        )

    baseline = baseline.model_copy(
        update={
            "process_local_identities": evidence(("a", "b")),
            "process_local_identities_truncated": True,
        }
    )
    same = eval_result_compatibility(baseline, baseline)
    assert same.comparable and not same.process_local_components_truncated
    current = baseline.model_copy(
        update={
            "accepted_exposure_comparison_revision": "sha256:" + "f" * 64,
            "process_local_identities": evidence(("b", "c")),
        }
    )
    result = eval_result_compatibility(baseline, current)
    assert result.reasons == ("accepted_exposure_contract_mismatch",)
    assert result.process_local_components == ()
    assert result.process_local_components_truncated
