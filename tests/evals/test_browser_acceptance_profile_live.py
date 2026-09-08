"""Opt-in canonical profile case through real Docker, WebBridge and virtual egress."""

import asyncio
import os
from dataclasses import replace

import pytest

from cayu import (
    AESGCMBrowserProfileKeyAuthority,
    BrowserProfileBinding,
    BrowserProfileDestinationPolicy,
    BrowserProfileScope,
    InMemoryBrowserProfileStore,
    SQLiteBrowserProfileStore,
)
from cayu.evals.browser_acceptance import BrowserAcceptanceManifestV1, run_browser_acceptance
from cayu.evals.browser_acceptance_fixture import BrowserAcceptanceFixtureV1
from cayu.evals.internal.browser_acceptance import (
    BrowserAcceptanceDeterministicProvider,
    _browser_results,
    build,
)
from cayu.evals.runner import EvalPlan
from cayu.tools.browser_session import (
    BROWSER_SESSION_PROTOCOL_VERSION,
    BROWSER_SESSION_WORKER_VERSION,
)


@pytest.mark.skipif(
    os.environ.get("CAYU_BROWSER_ACCEPTANCE_PROFILE_LIVE") != "1",
    reason="Requires the already prepared pinned Docker browser workload.",
)
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_real_profile_restores_cookie_into_a_new_browser(tmp_path, caplog, capfd, recwarn, backend):
    async def scenario(fixture):
        profile_store = (
            SQLiteBrowserProfileStore(tmp_path / "profiles.sqlite", store_id="profiles")
            if backend == "sqlite"
            else InMemoryBrowserProfileStore(store_id="profiles")
        )
        binding = BrowserProfileBinding.build(
            scope=BrowserProfileScope.build(
                application_id="browser-acceptance", tenant_id="fixture", sharing_scope="trial"
            ),
            destination_policy=BrowserProfileDestinationPolicy.build(
                ("https://docs.browser.test",)
            ),
            browser_protocol=BROWSER_SESSION_PROTOCOL_VERSION,
            browser_worker_version=BROWSER_SESSION_WORKER_VERSION,
            store=profile_store,
            key_authority=AESGCMBrowserProfileKeyAuthority(
                authority_id="trial-key", key=os.urandom(32)
            ),
        )
        app = None
        try:
            canonical = await build(fixture, profile_binding=binding)
            app = canonical.eval_plan.app
            suite = canonical.eval_plan.suite
            assert app is not None and suite is not None
            case = next(
                case
                for case in canonical.manifest.cases
                if case.case_id == "profile-cookie-restoration"
            )
            executable = next(
                case for case in suite.cases if case.id == "profile-cookie-restoration"
            )
            # A focused canonical row, not a claim that the entire corpus ran.
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
                eval_plan=EvalPlan(app=app, suite=suite.model_copy(update={"cases": [executable]})),
            )
            report = await run_browser_acceptance(plan, deterministic_fixture=fixture)
            assert "cayu_fixture_session" not in report.model_dump_json()
            row = report.rows[0]
            assert row.observed_state.value == "passed", row.model_dump_json()
            assert row.diagnostic.profile is not None
            proof = row.diagnostic.profile
            assert proof.store_kind == backend
            assert proof.generation_before == 0 and proof.generation_after == 2
            assert proof.checkpoint_receipt_revision and proof.restore_receipt_revision
            assert len({item.browser_session_revision for item in row.diagnostic.operations}) == 2
            provider = app.get_provider("browser-acceptance-scripted")
            assert type(provider) is BrowserAcceptanceDeterministicProvider
            results = _browser_results(provider.requests[-1])
            assert len(results) == 4
            assert results[0]["session_id"] != results[2]["session_id"]
            inspected = await profile_store.inspect_profile(binding.access)
            assert inspected.generation == 2 and not inspected.active_writer
            assert inspected.cookie_count == 1
            assert inspected.last_restore_receipt_id and inspected.last_checkpoint_receipt_id
            assert fixture.authenticated_request_count() == 1
        finally:
            if app is not None:
                assert await app.drain_environment_cleanups(timeout_s=15)
            if isinstance(profile_store, SQLiteBrowserProfileStore):
                await profile_store.close()

    with BrowserAcceptanceFixtureV1() as fixture:
        asyncio.run(scenario(fixture))
    if backend == "sqlite":
        assert b"cayu_fixture_session" not in (tmp_path / "profiles.sqlite").read_bytes()
    captured = capfd.readouterr()
    diagnostics = (
        captured.out + captured.err + caplog.text + str([item.message for item in recwarn])
    )
    assert "cayu_fixture_session" not in diagnostics
