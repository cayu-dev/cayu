"""Case application selection must agree with registration, identity and dispatch."""

import asyncio
from dataclasses import replace

import pytest
from tests.evals.test_browser_acceptance_execution import _plan

from cayu import ScriptedModelProvider, SQLiteSessionStore
from cayu.evals.browser_acceptance import (
    BrowserAcceptanceManifestV1,
    inspect_browser_acceptance_runtime_identity,
    run_browser_acceptance,
)
from cayu.evals.browser_acceptance_fixture import BrowserAcceptanceFixtureV1
from cayu.evals.internal.browser_acceptance import _ENVIRONMENT_EXECUTION_PROFILE_IDENTITY, build
from cayu.evals.runner import EvalPlan


def _provider(plan, name="scripted"):
    app = plan.eval_plan.app
    assert app is not None
    provider = app.get_provider(name)
    assert isinstance(provider, ScriptedModelProvider)
    return provider


def test_selected_case_app_owns_real_trial_and_identity(tmp_path):
    async def scenario(fixture):
        base, selected = (
            _plan(
                tmp_path / name,
                fixture,
                environment_profile_identity=_ENVIRONMENT_EXECUTION_PROFILE_IDENTITY,
            )
            for name in ("base", "selected")
        )
        plan = replace(
            base,
            case_apps=(("navigation", selected.eval_plan.app),),
            case_bridges=(("navigation", selected.bridge),),
        )
        identity = await inspect_browser_acceptance_runtime_identity(plan)
        report = await run_browser_acceptance(plan, deterministic_fixture=fixture)
        assert report.rows[0].conformance.value == "passed"
        assert (
            report.runtime_identity.execution_profile_fingerprint
            == identity.execution_profile_fingerprint
        )
        assert not _provider(base).requests
        assert len(_provider(selected).requests) == 2

    with BrowserAcceptanceFixtureV1() as fixture:
        asyncio.run(scenario(fixture))


@pytest.mark.parametrize("kind", ["unknown", "duplicate", "wrong_type", "unregistered_bridge"])
def test_invalid_case_app_refuses_before_dispatch(tmp_path, kind):
    async def scenario(fixture):
        base = _plan(tmp_path / "base", fixture)
        alternate = _plan(tmp_path / "alternate", fixture)
        app = alternate.eval_plan.app
        bindings = (("navigation", app),)
        if kind == "unknown":
            bindings = (("unknown", app),)
        elif kind == "duplicate":
            bindings = bindings * 2
        elif kind == "wrong_type":
            bindings = (("navigation", object()),)
        with pytest.raises(ValueError):
            plan = replace(base, case_apps=bindings)
            await run_browser_acceptance(plan, deterministic_fixture=fixture)
        assert not _provider(alternate).requests
        assert not _provider(base).requests

    with BrowserAcceptanceFixtureV1() as fixture:
        asyncio.run(scenario(fixture))


def test_missing_operator_binding_is_unavailable_without_dispatch():
    async def scenario(fixture):
        canonical = await build(fixture)
        app, suite = canonical.eval_plan.app, canonical.eval_plan.suite
        assert app is not None and suite is not None
        try:
            case = next(
                item
                for item in canonical.manifest.cases
                if item.case_id == "operator-private-handoff"
            )
            manifest = BrowserAcceptanceManifestV1.build(
                **{
                    **canonical.manifest.model_dump(mode="python", exclude={"revision"}),
                    "limits": canonical.manifest.limits,
                    "cases": (case,),
                }
            )
            plan = replace(
                canonical,
                manifest=manifest,
                case_bridges=(),
                eval_plan=EvalPlan(
                    app=app,
                    suite=suite.model_copy(
                        update={"cases": [item for item in suite.cases if item.id == case.case_id]}
                    ),
                ),
            )
            report = await run_browser_acceptance(plan, deterministic_fixture=fixture)
            row = report.rows[0]
            assert row.conformance.value == "incomplete"
            assert row.diagnostic.error_code == "operator_configuration_unavailable"
            assert not _provider(canonical, "browser-acceptance-scripted").requests
            assert fixture.request_counts() == {}
        finally:
            store = app.session_store
            assert isinstance(store, SQLiteSessionStore)
            await store.close()

    with BrowserAcceptanceFixtureV1() as fixture:
        asyncio.run(scenario(fixture))
