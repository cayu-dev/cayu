"""Controller support using the same protected setup as the acceptance CLI."""

import argparse
import asyncio
import os
import sys
from dataclasses import replace
from pathlib import Path

PRIVATE_CANARY = "browser-acceptance-private-operator-canary"


async def _controller():
    from pydantic import SecretStr

    from cayu.evals.browser_acceptance import BrowserAcceptanceManifestV1, run_browser_acceptance
    from cayu.evals.browser_acceptance_fixture import BrowserAcceptanceFixtureV1
    from cayu.evals.internal.browser_acceptance import BrowserAcceptanceDeterministicProvider, build
    from cayu.evals.internal.browser_acceptance_operator_server import operator_fixture_setup
    from cayu.evals.runner import EvalPlan

    proof = Path(os.environ["CAYU_OPERATOR_PROOF"])
    async with asyncio.timeout(10):
        while not (proof / "controller-id").exists():
            await asyncio.sleep(0.025)
    if os.environ.get("CAYU_BROWSER_ACCEPTANCE_FULL_CORPUS") == "1":
        import scripts.run_browser_acceptance as command

        os.environ["CAYU_BROWSER_ACCEPTANCE_CONTROL_CONTAINER"] = (
            (proof / "controller-id").read_text().strip()
        )
        os.environ["CAYU_BROWSER_ACCEPTANCE_CONTROL_CERTIFICATE"] = str(
            proof / "trust" / "control.crt"
        )
        os.environ["CAYU_BROWSER_ACCEPTANCE_CONTROL_PRIVATE_KEY"] = str(proof / "server.key")
        status = await command._run(
            argparse.Namespace(
                mode="deterministic",
                target=None,
                operator_setup=(
                    "cayu.evals.internal.browser_acceptance_operator_server:configured_fixture"
                ),
                output_directory=proof / "reports",
            )
        )
        reports = list((proof / "reports").glob("*.json"))
        assert len(reports) == 1
        (proof / "report.json").write_bytes(reports[0].read_bytes())
        assert status == 0, "Canonical CLI acceptance was incomplete or failed."
        print("PASS: complete canonical CLI browser corpus")
        return
    async with operator_fixture_setup(
        server_container_id=(proof / "controller-id").read_text().strip(),
        ca_certificate=proof / "trust" / "control.crt",
        server_private_key=proof / "server.key",
        private_text=SecretStr(PRIVATE_CANARY),
    ) as setup:
        with BrowserAcceptanceFixtureV1() as fixture:
            canonical = await build(fixture, operator_fixture=setup.binding)
            app, suite = canonical.eval_plan.app, canonical.eval_plan.suite
            assert app is not None and suite is not None
            selected_ids = {
                "operator-private-handoff",
                "visual-canvas-control",
                "page-popup-process-loss-ambiguity",
                "action-detached-control",
                "action-hover-detached",
                "action-replaced-element",
                "artifact-upload-disconnection",
                "artifact-upload-acknowledgement-loss",
                "visual-process-terminal-replay",
            }
            manifest = BrowserAcceptanceManifestV1.build(
                **{
                    **canonical.manifest.model_dump(mode="python", exclude={"revision"}),
                    "limits": canonical.manifest.limits,
                    "cases": tuple(
                        item for item in canonical.manifest.cases if item.case_id in selected_ids
                    ),
                }
            )
            plan = replace(
                canonical,
                manifest=manifest,
                case_bridges=tuple(
                    item for item in canonical.case_bridges if item[0] in selected_ids
                ),
                case_apps=tuple(item for item in canonical.case_apps if item[0] in selected_ids),
                eval_plan=EvalPlan(
                    app=app,
                    suite=suite.model_copy(
                        update={"cases": [item for item in suite.cases if item.id in selected_ids]}
                    ),
                ),
            )
            async with setup.serve(plan):
                async with asyncio.timeout(180):
                    report = await run_browser_acceptance(plan, deterministic_fixture=fixture)
                encoded = report.model_dump_json()
                assert PRIVATE_CANARY not in encoded
                (proof / "report.json").write_text(encoded)
                assert all(item.conformance.value == "passed" for item in report.rows), encoded
                row = next(
                    item for item in report.rows if item.case_id == "operator-private-handoff"
                )
                assert row.observed_state.value == "passed", row.model_dump_json()
                assert row.semantic_state.value == "passed", row.model_dump_json()
                assert row.diagnostic.operator is not None
                assert row.diagnostic.operator.state == "closed"
                assert row.diagnostic.operator.fresh_observation_revision is not None
                assert row.diagnostic.fixture_effects == {"operator-input": 1}
                provider = app.get_provider("browser-acceptance-scripted")
                assert isinstance(provider, BrowserAcceptanceDeterministicProvider)
                assert PRIVATE_CANARY not in repr(provider.requests)
            print("PASS: canonical protected Docker operator handoff")


if __name__ == "__main__" and sys.argv[1:] == ["controller"]:
    asyncio.run(_controller())
