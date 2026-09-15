from __future__ import annotations

import asyncio

import pytest

from cayu.evals.benchmark_campaign import (
    BenchmarkCampaignSettingsV1,
    admit_benchmark_campaign,
    campaign_run_request,
    execute_benchmark_campaign,
    load_benchmark_campaign,
    prepare_benchmark_campaign,
    resume_benchmark_campaign,
)
from cayu.evals.benchmark_comparison import compare_benchmark_campaigns
from cayu.evals.benchmark_inspection import export_benchmark_campaign, inspect_benchmark_campaign
from cayu.evals.benchmark_package import (
    BenchmarkPackageV1,
    benchmark_package_to_json,
    load_benchmark_package,
)
from cayu.evals.benchmark_rescore import rescore_benchmark_campaign
from cayu.evals.benchmark_retry import retry_benchmark_trial
from cayu.evals.benchmark_synthetic import (
    build_synthetic_benchmark_plan,
    write_synthetic_benchmark_package,
)
from cayu.evals.runner import EvalPlan
from cayu.evals.store import EvalRunClaimLost, EvalRunRecoveryPolicyV1
from cayu.storage.evals_sqlite import SQLiteEvalStore
from cayu.storage.migrations import SchemaMode


def test_interrupted_campaign_preserves_checkpoint_and_blocks_replay(tmp_path, monkeypatch):
    async def scenario():
        loaded = load_benchmark_package(write_synthetic_benchmark_package(tmp_path / "package"))
        plan = build_synthetic_benchmark_plan(tmp_path / "target")
        target = plan.corpus_target
        directory = tmp_path / "campaign"
        prepared = await prepare_benchmark_campaign(
            loaded,
            plan,
            case_ids=["echo", "wrong-answer"],
            settings=BenchmarkCampaignSettingsV1(max_retry_attempts=1),
        )
        campaign = await admit_benchmark_campaign(prepared, directory)
        store = SQLiteEvalStore(directory / "evals.sqlite3")
        saved = asyncio.Event()
        original = SQLiteEvalStore.save_trial_checkpoint

        async def checkpoint(self, *args, **kwargs):
            await original(self, *args, **kwargs)
            saved.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(SQLiteEvalStore, "save_trial_checkpoint", checkpoint)
        task = asyncio.create_task(
            execute_benchmark_campaign(campaign, directory, registry=prepared.registry, store=store)
        )
        try:
            await asyncio.wait_for(saved.wait(), timeout=30)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            calls = len(target.app.get_provider().requests)
            assert calls == 1
            monkeypatch.setattr(SQLiteEvalStore, "save_trial_checkpoint", original)
            await resume_benchmark_campaign(directory, plan)
            inspection = await inspect_benchmark_campaign(directory)
            assert len(target.app.get_provider().requests) == calls
            assert [(row.case_id, row.status) for row in inspection.trials] == [
                ("echo", "passed"),
                ("wrong-answer", "unavailable"),
            ]
            assert inspection.trials[1].diagnostic_code == "recovery_reexecution_blocked"
            successor = await retry_benchmark_trial(
                directory, plan, case_id="wrong-answer", trial_number=1
            )
            assert len(target.app.get_provider().requests) == calls + 1
            assert (
                load_benchmark_campaign(successor).retry_of.source_trial_revision
                == inspection.trials[1].source_trial_revision
            )
            await retry_benchmark_trial(directory, plan, case_id="wrong-answer", trial_number=1)
            assert len(target.app.get_provider().requests) == calls + 1
            with pytest.raises(ValueError, match="allowance"):
                await retry_benchmark_trial(
                    directory, plan, case_id="wrong-answer", trial_number=1, attempt=2
                )
        finally:
            if not task.done():
                task.cancel()
            await store.close()
            await target.app.session_store.close()

    asyncio.run(scenario())


def test_comparison_and_static_rescore_never_dispatch(tmp_path):
    async def scenario():
        loaded = load_benchmark_package(write_synthetic_benchmark_package(tmp_path / "package"))
        plan = build_synthetic_benchmark_plan(tmp_path / "target")
        directory = tmp_path / "campaign"
        prepared = await prepare_benchmark_campaign(loaded, plan)
        await admit_benchmark_campaign(prepared, directory)
        try:
            await resume_benchmark_campaign(directory, plan)
            calls = len(plan.corpus_target.app.get_provider().requests)
            comparison = await compare_benchmark_campaigns(directory, directory)
            assert comparison["compatibility"] == "comparable"
            assert comparison["regressions"] == 0
            assert len(comparison["trials"]) == 3
            scorer_path = write_synthetic_benchmark_package(tmp_path / "scorer")
            package = loaded.package
            scorer = BenchmarkPackageV1.create(
                id=package.id,
                version=package.version,
                scorer_id=package.scorer_id,
                scorer_version="2",
                suite=package.suite,
                scenarios=package.scenarios,
                files=package.files,
                requirements=package.requirements,
            )
            scorer_path.write_text(benchmark_package_to_json(scorer))
            rescored = await rescore_benchmark_campaign(directory, scorer_path)
            assert len(rescored["trials"]) == 3
            assert [
                assertion["outcome"]
                for row in rescored["trials"]
                for assertion in row["assertions"]
            ].count("failed") == 1
            assert rescored["judge_calls"] == rescored["candidate_calls"] == 0
            assert len(plan.corpus_target.app.get_provider().requests) == calls
            other = await prepare_benchmark_campaign(loaded, plan, case_ids=["echo"])
            await admit_benchmark_campaign(other, tmp_path / "other")
            mismatch = await compare_benchmark_campaigns(directory, tmp_path / "other")
            assert mismatch["compatibility"] == "incompatible"
            assert "cohort" in mismatch["mismatches"]
            assert mismatch["trials"] == []
        finally:
            await plan.corpus_target.app.session_store.close()

    asyncio.run(scenario())


def test_typed_failures_selective_retry_and_retained_native_links(tmp_path):
    async def scenario():
        loaded = load_benchmark_package(
            write_synthetic_benchmark_package(tmp_path / "package", failure_modes=True)
        )
        plan = build_synthetic_benchmark_plan(tmp_path / "target")
        directory = tmp_path / "campaign"
        prepared = await prepare_benchmark_campaign(
            loaded, plan, settings=BenchmarkCampaignSettingsV1(max_retry_attempts=1)
        )
        await admit_benchmark_campaign(prepared, directory)
        try:
            await resume_benchmark_campaign(directory, plan)
            original = await inspect_benchmark_campaign(directory, include_sessions=True)
            categories = {row.case_id: row.failure_category for row in original.trials}
            assert categories["provider-failure"] == "provider_failure"
            assert categories["scoring-failure"] == "scoring_failure"
            assert categories["wrong-answer"] == "answer_mismatch"
            with pytest.raises(ValueError, match="eligible"):
                await retry_benchmark_trial(directory, plan, case_id="wrong-answer", trial_number=1)
            child = await retry_benchmark_trial(
                directory, plan, case_id="provider-failure", trial_number=1
            )
            successor = await inspect_benchmark_campaign(child)
            assert successor.status == "passed"
            latest = await inspect_benchmark_campaign(directory)
            assert len(latest.successors) == 1
            assert latest.successors[0]["observed_total_tokens"] == "10"
            assert (
                next(row for row in latest.trials if row.case_id == "provider-failure").status
                == "error"
            )
            store = SQLiteEvalStore(
                directory / "evals.sqlite3", read_only=True, schema_mode=SchemaMode.VALIDATE
            )
            try:
                for run in prepared.campaign.runs:
                    links = await store.load_trial_evidence_links(run.spec.id)
                    assert len(links) == len(run.case_ids)
                    result = await store.load_result(run.spec.id)
                    assert all(link.session_id not in result.model_dump_json() for link in links)
            finally:
                await store.close()
        finally:
            await plan.corpus_target.app.session_store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("prior_claims,allowance,expected_calls", [(1, 1, 0), (1, 2, 1), (2, 2, 0)])
def test_recovery_attempt_bounds_and_competing_resume(
    tmp_path, prior_claims, allowance, expected_calls
):
    async def scenario():
        loaded = load_benchmark_package(write_synthetic_benchmark_package(tmp_path / "package"))
        plan = build_synthetic_benchmark_plan(tmp_path / "target")
        directory = tmp_path / "campaign"
        settings = BenchmarkCampaignSettingsV1(
            recovery_policy=EvalRunRecoveryPolicyV1(
                mode="caller_authorized" if allowance > 1 else "checkpoint_only",
                max_execution_attempts=allowance,
            )
        )
        prepared = await prepare_benchmark_campaign(
            loaded, plan, settings=settings, case_ids=["echo"]
        )
        await admit_benchmark_campaign(prepared, directory)
        store = SQLiteEvalStore(directory / "evals.sqlite3")
        try:
            for _ in range(prior_claims):
                lease = await store.claim_run(launch_revision=prepared.campaign.launch_revision)
                assert lease is not None
                await store.release_run(lease.claim)
            await asyncio.gather(
                resume_benchmark_campaign(directory, plan),
                resume_benchmark_campaign(directory, plan),
            )
            assert len(plan.corpus_target.app.get_provider().requests) == expected_calls
            with pytest.raises(EvalRunClaimLost):
                await store.heartbeat_run(lease.claim, extend_seconds=30)
            result = await inspect_benchmark_campaign(directory)
            assert result.status == ("passed" if expected_calls else "incomplete")
            cancelled = await prepare_benchmark_campaign(loaded, plan, case_ids=["echo"])
            cancel_directory = tmp_path / "cancelled"
            await admit_benchmark_campaign(cancelled, cancel_directory)
            cancel_store = SQLiteEvalStore(cancel_directory / "evals.sqlite3")
            try:
                await cancel_store.request_cancel(cancelled.campaign.runs[0].spec.id)
            finally:
                await cancel_store.close()
            await resume_benchmark_campaign(cancel_directory, plan)
            assert len(plan.corpus_target.app.get_provider().requests) == expected_calls
            assert (await inspect_benchmark_campaign(cancel_directory)).counts["cancelled"] == 1
        finally:
            await store.close()
            await plan.corpus_target.app.session_store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("trials,concurrency", [(1, 1), (2, 2)])
def test_native_campaign_launch_and_scoped_claims(tmp_path, trials, concurrency):
    async def scenario():
        loaded = load_benchmark_package(write_synthetic_benchmark_package(tmp_path / "package"))
        plan = build_synthetic_benchmark_plan(tmp_path / "target")
        target = plan.corpus_target
        assert target is not None
        try:
            prepared = await prepare_benchmark_campaign(
                loaded,
                plan,
                settings=BenchmarkCampaignSettingsV1(trials=trials, max_concurrency=concurrency),
            )
            assert not target.app.get_provider().requests
            campaign = await admit_benchmark_campaign(prepared, tmp_path / "campaign")
            assert load_benchmark_campaign(tmp_path / "campaign") == campaign
            assert not target.app.get_provider().requests
            store = SQLiteEvalStore(tmp_path / "campaign/evals.sqlite3")
            try:
                # Another launch sharing this target must never be picked up by these workers.
                request = campaign_run_request(campaign, campaign.runs[0])
                unrelated = request.model_copy(
                    update={
                        "run_id": "unrelated",
                        "idempotency_key": "sha256:" + "e" * 64,
                        "invocation": request.invocation.model_copy(
                            update={"authored_suite_launch_revision": "sha256:" + "f" * 64}
                        ),
                    }
                )
                await store.admit_run(unrelated, redact_json=target.app.redact_json)
                async with asyncio.timeout(45):
                    await execute_benchmark_campaign(
                        campaign, tmp_path / "campaign", registry=prepared.registry, store=store
                    )
                assert (await store.load_run("unrelated")).status == "queued"
                outcomes = {}
                for run in campaign.runs:
                    record = await store.load_run(run.spec.id)
                    assert record.status == "completed", record
                    result = await store.load_result(run.spec.id)
                    assert result is not None
                    for case in result.run.cases:
                        assert len(case.trials) == trials
                        outcomes[case.case_id] = case.status
                assert outcomes == {
                    "echo": "passed",
                    "wrong-answer": "failed",
                    "attachment": "passed",
                }
                assert len(target.app.get_provider().requests) == 3 * trials
                inspection = await inspect_benchmark_campaign(
                    tmp_path / "campaign", include_sessions=True
                )
                assert inspection.counts["total"] == 3 * trials
                assert inspection.status == "failed"
                assert all(
                    row.session_association == "result_revision" for row in inspection.trials
                ), inspection
                assert all(row.session_evidence is not None for row in inspection.trials)
                assert all(row.usage_availability == "known" for row in inspection.trials)
                await export_benchmark_campaign(tmp_path / "campaign", tmp_path / "export.zip")
                assert len(target.app.get_provider().requests) == 3 * trials
            finally:
                await store.close()
        finally:
            await target.app.session_store.close()

    asyncio.run(scenario())


def test_concurrency_requires_existing_target_isolation_contract(tmp_path):
    async def scenario():
        loaded = load_benchmark_package(write_synthetic_benchmark_package(tmp_path / "package"))
        plan = build_synthetic_benchmark_plan(tmp_path / "target")
        target = plan.corpus_target
        assert target is not None
        try:
            undeclared = EvalPlan(corpus_target=target)
            with pytest.raises(ValueError, match="declared execution profile"):
                await prepare_benchmark_campaign(
                    loaded, undeclared, settings=BenchmarkCampaignSettingsV1(max_concurrency=2)
                )
            assert not target.app.get_provider().requests
        finally:
            await target.app.session_store.close()

    asyncio.run(scenario())


def test_package_subset_retains_valid_catalog_scenario_dependencies(tmp_path):
    async def scenario():
        loaded = load_benchmark_package(write_synthetic_benchmark_package(tmp_path / "package"))
        plan = build_synthetic_benchmark_plan(tmp_path / "target")
        try:
            prepared = await prepare_benchmark_campaign(loaded, plan, case_ids=["echo"])
            await admit_benchmark_campaign(prepared, tmp_path / "campaign")
            assert prepared.campaign.runs[0].case_ids == ("echo",)
        finally:
            await plan.corpus_target.app.session_store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "minimum,override,expected", [(1, True, "passed"), (1, False, "passed"), (2, False, "failed")]
)
def test_campaign_honors_trial_threshold_online_and_offline(
    tmp_path, monkeypatch, minimum, override, expected
):
    from dataclasses import replace

    from cayu.evals.benchmark_inspection import load_benchmark_export
    from cayu.evals.benchmark_synthetic import ModelStreamEvent, _SyntheticBenchmarkProvider
    from cayu.evals.suite_authoring import (
        EvalSuiteDraftV3,
        EvalSuiteTrialRequestDraftV3,
        compile_eval_suite_draft_v3,
    )

    calls = 0

    def respond(self, request):
        nonlocal calls
        calls += 1
        return (
            ModelStreamEvent.text_delta("READY" if calls == 1 else "WRONG"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        )

    monkeypatch.setattr(_SyntheticBenchmarkProvider, "_respond", respond)

    async def scenario():
        loaded = load_benchmark_package(write_synthetic_benchmark_package(tmp_path / "package"))
        if not override:
            draft = EvalSuiteDraftV3.from_document(loaded.package.suite)
            suite = compile_eval_suite_draft_v3(
                draft.model_copy(
                    update={
                        "trial_request": EvalSuiteTrialRequestDraftV3(
                            trials=2, minimum_passed_trials=minimum
                        )
                    }
                )
            )
            package = loaded.package
            loaded = replace(
                loaded,
                package=BenchmarkPackageV1.create(
                    id=package.id,
                    version=package.version,
                    scorer_id=package.scorer_id,
                    scorer_version=package.scorer_version,
                    suite=suite,
                    scenarios=package.scenarios,
                    files=package.files,
                    requirements=package.requirements,
                ),
            )
        plan = build_synthetic_benchmark_plan(tmp_path / "target")
        directory = tmp_path / "campaign"
        try:
            settings = (
                BenchmarkCampaignSettingsV1(trials=2, minimum_passed_trials=minimum)
                if override
                else None
            )
            prepared = await prepare_benchmark_campaign(
                loaded, plan, case_ids=["echo"], settings=settings
            )
            await admit_benchmark_campaign(prepared, directory)
            await resume_benchmark_campaign(directory, plan)
            store = SQLiteEvalStore(directory / "evals.sqlite3")
            try:
                result = await store.load_result(prepared.campaign.runs[0].spec.id)
                assert result.run.status == expected
            finally:
                await store.close()
            inspection = await inspect_benchmark_campaign(directory)
            assert inspection.status == expected
            assert [row.status for row in inspection.trials] == ["passed", "failed"]
            exported = tmp_path / "export.zip"
            await export_benchmark_campaign(directory, exported)
            assert load_benchmark_export(exported).status == expected
            assert calls == 2
        finally:
            await plan.corpus_target.app.session_store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("interruption", ["receipt", "publication"])
def test_retry_admission_interruption_and_competing_publishers(tmp_path, monkeypatch, interruption):
    import cayu.evals.benchmark_campaign as campaign_module

    async def scenario():
        loaded = load_benchmark_package(
            write_synthetic_benchmark_package(tmp_path / "package", failure_modes=True)
        )
        plan = build_synthetic_benchmark_plan(tmp_path / "target")
        directory = tmp_path / "campaign"
        try:
            prepared = await prepare_benchmark_campaign(
                loaded,
                plan,
                case_ids=["provider-failure"],
                settings=BenchmarkCampaignSettingsV1(max_retry_attempts=1),
            )
            await admit_benchmark_campaign(prepared, directory)
            await resume_benchmark_campaign(directory, plan)
            original_write = campaign_module.write_process_document
            original_publish = campaign_module._rename_directory_no_replace

            def fail_receipt(path, value):
                if path.name == "campaign.json":
                    raise RuntimeError("interrupted admission")
                return original_write(path, value)

            def fail_publication(*args, **kwargs):
                raise RuntimeError("interrupted admission")

            if interruption == "receipt":
                monkeypatch.setattr(campaign_module, "write_process_document", fail_receipt)
            else:
                monkeypatch.setattr(
                    campaign_module, "_rename_directory_no_replace", fail_publication
                )
            with pytest.raises(RuntimeError, match="interrupted admission"):
                await retry_benchmark_trial(
                    directory, plan, case_id="provider-failure", trial_number=1
                )
            monkeypatch.setattr(campaign_module, "write_process_document", original_write)
            monkeypatch.setattr(campaign_module, "_rename_directory_no_replace", original_publish)
            assert len(plan.corpus_target.app.get_provider().requests) == 1

            # Force both writers to finish staging before either can publish.
            original_stage = campaign_module._write_benchmark_campaign
            staged = 0
            both_staged = asyncio.Event()

            async def stage(*args, **kwargs):
                nonlocal staged
                await original_stage(*args, **kwargs)
                staged += 1
                if staged == 2:
                    both_staged.set()
                await asyncio.wait_for(both_staged.wait(), timeout=30)

            monkeypatch.setattr(campaign_module, "_write_benchmark_campaign", stage)
            successors = await asyncio.gather(
                *(
                    retry_benchmark_trial(
                        directory, plan, case_id="provider-failure", trial_number=1
                    )
                    for _ in range(2)
                )
            )
            assert successors[0] == successors[1]
            assert (await inspect_benchmark_campaign(successors[0])).status == "passed"
            await retry_benchmark_trial(directory, plan, case_id="provider-failure", trial_number=1)
            assert len(plan.corpus_target.app.get_provider().requests) == 2
            assert len((await inspect_benchmark_campaign(directory)).successors) == 1
        finally:
            await plan.corpus_target.app.session_store.close()

    asyncio.run(scenario())
