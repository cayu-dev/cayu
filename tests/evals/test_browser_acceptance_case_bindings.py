"""Case-local browser configuration remains bound to registered execution authority."""

import asyncio
from dataclasses import replace

import pytest
from tests.evals.test_browser_acceptance_execution import _plan

from cayu import AgentSpec, WebBridge
from cayu.evals.browser_acceptance import (
    BrowserAcceptanceCaseV1,
    BrowserAcceptanceManifestV1,
    _portable_execution_value,
    inspect_browser_acceptance_runtime_identity,
    run_browser_acceptance,
)
from cayu.evals.browser_acceptance_fixture import BrowserAcceptanceFixtureV1
from cayu.evals.internal.browser_acceptance import _ENVIRONMENT_EXECUTION_PROFILE_IDENTITY
from cayu.evals.runner import EvalPlan
from cayu.runners import PINNED_BROWSER_SESSION_WORKLOAD


@pytest.mark.parametrize("timeout", [float("inf"), float("-inf"), float("nan")])
def test_case_profile_authority_rejects_nonfinite_timeout(timeout):
    with pytest.raises(ValueError, match="unsupported execution material"):
        _portable_execution_value({"limits": {"timeout": timeout}}, "case authority")


def test_case_profile_authority_retains_fractional_timeout():
    value = {"limits": {"timeout": 0.25}}
    assert _portable_execution_value(value, "case authority") == value


def _bound_plan(source, *, max_operations=4, max_artifact_bytes=1 << 19):
    app, suite = source.eval_plan.app, source.eval_plan.suite
    bridge = WebBridge.sandboxed_browser(
        environment=app.get_environment_factory("browser"),
        browser_image=PINNED_BROWSER_SESSION_WORKLOAD.image,
        interactive=True,
        interactive_options={
            "max_operations": max_operations,
            "max_artifact_bytes": max_artifact_bytes,
        },
    )
    bridge.register_agent(app, AgentSpec(name="case-agent", model="scripted-browser-v1"))
    case = suite.cases[0]
    case = case.model_copy(
        update={"request": case.request.model_copy(update={"agent_name": "case-agent"})}
    )
    return replace(
        source,
        eval_plan=EvalPlan(app=app, suite=suite.model_copy(update={"cases": [case]})),
        case_bridges=((case.id, bridge),),
    )


def test_case_binding_runs_registered_browser_and_changes_suite_identity(tmp_path):
    with BrowserAcceptanceFixtureV1() as fixture:
        source = _plan(
            tmp_path, fixture, environment_profile_identity=_ENVIRONMENT_EXECUTION_PROFILE_IDENTITY
        )
        before = asyncio.run(inspect_browser_acceptance_runtime_identity(source))
        bound = _bound_plan(source)
        after = asyncio.run(inspect_browser_acceptance_runtime_identity(bound))
        assert before.execution_suite_fingerprint != after.execution_suite_fingerprint
        report = asyncio.run(run_browser_acceptance(bound, deterministic_fixture=fixture))
        assert report.rows[0].observed_state.value == "passed"
        assert (
            report.runtime_identity.execution_profile_fingerprint
            == after.execution_profile_fingerprint
        )


@pytest.mark.parametrize(
    "invalid", ["unknown", "duplicate", "unregistered", "operation_limit", "artifact_limit"]
)
def test_invalid_case_binding_rejects_before_provider_dispatch(tmp_path, invalid):
    with BrowserAcceptanceFixtureV1() as fixture:
        source = _plan(
            tmp_path, fixture, environment_profile_identity=_ENVIRONMENT_EXECUTION_PROFILE_IDENTITY
        )
        provider = source.eval_plan.app.get_provider("scripted")
        with pytest.raises(ValueError):
            bound = _bound_plan(
                source,
                max_operations=5 if invalid == "operation_limit" else 4,
                max_artifact_bytes=2 << 20 if invalid == "artifact_limit" else 1 << 19,
            )
            key, bridge = bound.case_bridges[0]
            if invalid == "unknown":
                bound = replace(bound, case_bridges=(("missing-case", bridge),))
            elif invalid == "duplicate":
                bound = replace(bound, case_bridges=((key, bridge), (key, bridge)))
            elif invalid == "unregistered":
                bound = replace(bound, case_bridges=((key, source.bridge),))
            asyncio.run(run_browser_acceptance(bound, deterministic_fixture=fixture))
        assert provider.requests == []


def test_distinct_case_limits_cannot_understate_combined_artifact_budget(tmp_path):
    with BrowserAcceptanceFixtureV1() as fixture:
        source = _plan(
            tmp_path, fixture, environment_profile_identity=_ENVIRONMENT_EXECUTION_PROFILE_IDENTITY
        )
        bound = _bound_plan(source)
        first_manifest = source.manifest.cases[0]
        second_manifest = BrowserAcceptanceCaseV1.build(
            **{
                **first_manifest.model_dump(mode="python", exclude={"revision"}),
                "case_id": "second-navigation",
            }
        )
        second_case = bound.eval_plan.suite.cases[0].model_copy(
            update={
                "id": second_manifest.case_id,
                "metadata": {"browser_acceptance_case_revision": second_manifest.revision},
            }
        )
        manifest = BrowserAcceptanceManifestV1.build(
            **{
                **source.manifest.model_dump(mode="python", exclude={"revision"}),
                "cases": (first_manifest, second_manifest),
                "limits": source.manifest.limits.model_copy(
                    update={
                        "max_browser_operations": 8,
                        "max_model_steps": 4,
                        "max_artifact_bytes": 5 << 20,
                    }
                ),
            }
        )
        plan = replace(
            source,
            manifest=manifest,
            eval_plan=EvalPlan(
                app=source.eval_plan.app,
                suite=source.eval_plan.suite.model_copy(
                    update={"cases": [source.eval_plan.suite.cases[0], second_case]}
                ),
            ),
            case_bridges=((second_case.id, bound.case_bridges[0][1]),),
        )
        # Each case fits 5 MiB, but 4*1 MiB + 4*0.5 MiB exceeds it. Using
        # the final case's per-operation limit for every case would admit 4 MiB.
        with pytest.raises(ValueError, match="aggregate artifact ceiling"):
            asyncio.run(run_browser_acceptance(plan, deterministic_fixture=fixture))
        assert source.eval_plan.app.get_provider("scripted").requests == []
