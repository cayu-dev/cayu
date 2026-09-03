from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest

import cayu.evals.causal_memory_campaign as causal_memory_campaign
from cayu.evals.causal_memory_campaign import (
    CAUSAL_MEMORY_CAMPAIGN_REPETITIONS,
    CAUSAL_MEMORY_CAMPAIGN_VARIANTS,
    load_causal_memory_reference_corpus,
    run_causal_memory_reference_campaign,
)
from cayu.evals.memory_attribution import EvalMemoryEvidenceCompleteness
from cayu.evals.memory_reporting import (
    MemoryTrialAvailability,
    MemoryVariantDispositionStatus,
    memory_experiment_report_to_json,
    render_memory_experiment_report_html,
)
from cayu.memory_intervention_execution import (
    MemoryInterventionExecutionConflict,
    MemoryInterventionExecutionStatus,
    MemoryInterventionExecutor,
    MemoryInterventionTrialOutcome,
    MemoryInterventionTrialRequest,
)
from cayu.memory_interventions import MemoryInterventionKind
from cayu.providers import ModelRequest
from cayu.runtime.manifest import APP_MANIFEST_SCHEMA_VERSION

_ROOT = Path(__file__).resolve().parents[2]
_CORPUS = _ROOT / "benchmarks/memory/causal-memory-campaign-corpus-v1.json"
_SCRIPT = _ROOT / "scripts/run_causal_memory_reference_campaign.py"


class _StopAfterDurableTrials(RuntimeError):
    pass


def test_reference_campaign_runs_real_paired_trials_and_recovers_in_fresh_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = load_causal_memory_reference_corpus(_CORPUS)
    state_directory = tmp_path / "state"
    original_execute_trial = MemoryInterventionExecutor.execute_trial
    original_scripted_provider = causal_memory_campaign._scripted_provider
    original_validate_request = causal_memory_campaign._validate_runtime_request
    campaign_providers = []
    audited_trials: list[tuple[str, str]] = []
    durable_trials = 0

    def capture_provider(*, recover_only: bool):
        provider = original_scripted_provider(recover_only=recover_only)
        campaign_providers.append(provider)
        return provider

    monkeypatch.setattr(causal_memory_campaign, "_scripted_provider", capture_provider)

    def capture_request_audit(
        request: ModelRequest,
        *,
        case_id: str,
        variant_id: str,
    ) -> None:
        original_validate_request(request, case_id=case_id, variant_id=variant_id)
        audited_trials.append((case_id, variant_id))

    monkeypatch.setattr(
        causal_memory_campaign,
        "_validate_runtime_request",
        capture_request_audit,
    )

    async def stop_after_five_trials(
        executor: MemoryInterventionExecutor,
        request: MemoryInterventionTrialRequest,
    ) -> MemoryInterventionTrialOutcome:
        nonlocal durable_trials
        outcome = await original_execute_trial(executor, request)
        durable_trials += 1
        if durable_trials == 5:
            raise _StopAfterDurableTrials
        return outcome

    monkeypatch.setattr(MemoryInterventionExecutor, "execute_trial", stop_after_five_trials)
    with pytest.raises(_StopAfterDurableTrials):
        asyncio.run(run_causal_memory_reference_campaign(corpus, state_directory))
    assert durable_trials == 5
    assert len(audited_trials) == 5
    monkeypatch.setattr(MemoryInterventionExecutor, "execute_trial", original_execute_trial)

    with pytest.raises(
        MemoryInterventionExecutionConflict,
        match="every campaign trial to be terminal",
    ):
        asyncio.run(
            run_causal_memory_reference_campaign(corpus, state_directory, recover_only=True)
        )
    assert campaign_providers[-1].requests == []

    report = asyncio.run(run_causal_memory_reference_campaign(corpus, state_directory))

    assert {
        case.source.app_manifest_schema_version for case in corpus.cases if case.source is not None
    } == {APP_MANIFEST_SCHEMA_VERSION}
    assert report.selected_variant_id == "as-declared"
    assert len(report.rows) == (
        len(corpus.cases)
        * CAUSAL_MEMORY_CAMPAIGN_REPETITIONS
        * len(CAUSAL_MEMORY_CAMPAIGN_VARIANTS)
    )
    assert all(row.availability is MemoryTrialAvailability.AVAILABLE for row in report.rows)
    assert all(
        row.execution_status is MemoryInterventionExecutionStatus.COMPLETED for row in report.rows
    )
    assert all(
        row.attribution_status is EvalMemoryEvidenceCompleteness.COMPLETE for row in report.rows
    )
    assert {variant.variant_id: variant.spec.kind for variant in report.variants} == {
        "as-declared": MemoryInterventionKind.AS_DECLARED,
        "automatic-recall-off": MemoryInterventionKind.AUTOMATIC_RECALL_OFF,
        "omit-items": MemoryInterventionKind.OMIT_ITEMS,
        "replace-items": MemoryInterventionKind.REPLACE_ITEMS,
    }
    dispositions = {item.variant_id: item.status for item in report.dispositions}
    assert dispositions == {
        "as-declared": MemoryVariantDispositionStatus.SELECTED,
        "automatic-recall-off": MemoryVariantDispositionStatus.NOT_BETTER,
        "omit-items": MemoryVariantDispositionStatus.NOT_BETTER,
        "replace-items": MemoryVariantDispositionStatus.REJECTED,
    }
    published_statuses = {
        (case_id, variant_id): {
            row.published_status
            for row in report.rows
            if row.case_id == case_id and row.variant_id == variant_id
        }
        for case_id in (case.id for case in corpus.cases)
        for variant_id in CAUSAL_MEMORY_CAMPAIGN_VARIANTS
    }
    assert published_statuses == {
        ("cross-source-follow-up", "as-declared"): {"passed"},
        ("cross-source-follow-up", "automatic-recall-off"): {"passed"},
        ("cross-source-follow-up", "omit-items"): {"passed"},
        ("cross-source-follow-up", "replace-items"): {"failed"},
        ("helpful-current-memory", "as-declared"): {"passed"},
        ("helpful-current-memory", "automatic-recall-off"): {"failed"},
        ("helpful-current-memory", "omit-items"): {"failed"},
        ("helpful-current-memory", "replace-items"): {"failed"},
        ("neutral-authority-silence", "as-declared"): {"passed"},
        ("neutral-authority-silence", "automatic-recall-off"): {"passed"},
        ("neutral-authority-silence", "omit-items"): {"passed"},
        ("neutral-authority-silence", "replace-items"): {"passed"},
    }
    state_scope_ids = {
        row.intervention_binding.operation.state_scope_id
        for row in report.rows
        if row.intervention_binding is not None
    }
    assert len(state_scope_ids) == len(report.rows)
    assert any(
        row.variant_id == "replace-items"
        and row.case_id == "helpful-current-memory"
        and row.published_status == "failed"
        and row.availability is MemoryTrialAvailability.AVAILABLE
        for row in report.rows
    )
    assert "Complete trial matrix" in render_memory_experiment_report_html(report)

    replayed = asyncio.run(run_causal_memory_reference_campaign(corpus, state_directory))
    assert memory_experiment_report_to_json(replayed) == memory_experiment_report_to_json(report)
    dispatched_requests = [
        request for provider in campaign_providers for request in provider.requests
    ]
    assert len(dispatched_requests) == len(report.rows)
    assert len(audited_trials) == len(report.rows)
    assert sum(
        causal_memory_campaign._CONFLICTING_TEXT in causal_memory_campaign._request_text(request)
        for request in dispatched_requests
    ) == (2 * CAUSAL_MEMORY_CAMPAIGN_REPETITIONS)
    assert (
        sum(
            causal_memory_campaign._PRIMARY_CURRENT_TEXT
            in causal_memory_campaign._request_text(request)
            and "answer this instruction exactly: NEUTRAL."
            in causal_memory_campaign._request_text(request)
            for request in dispatched_requests
        )
        == CAUSAL_MEMORY_CAMPAIGN_REPETITIONS
    )

    recovered_output = tmp_path / "recovered.json"
    environment = os.environ.copy()
    existing_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(_ROOT / "src")
        if not existing_path
        else os.pathsep.join((str(_ROOT / "src"), existing_path))
    )
    recovered = subprocess.run(
        (
            sys.executable,
            str(_SCRIPT),
            "--corpus",
            str(_CORPUS),
            "--state-directory",
            str(state_directory),
            "--output",
            str(recovered_output),
            "--recover-only",
        ),
        cwd=_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert recovered.returncode == 0, recovered.stderr
    assert recovered.stdout == ""
    assert recovered_output.read_text(encoding="utf-8").rstrip("\n") == (
        memory_experiment_report_to_json(report)
    )


def test_reference_campaign_recovers_an_interrupted_trial_without_survivor_filtering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = load_causal_memory_reference_corpus(_CORPUS)
    state_directory = tmp_path / "state"
    interrupted_provider = causal_memory_campaign._CampaignScriptedProvider(recover_only=False)
    recovery_provider = causal_memory_campaign._CampaignScriptedProvider(recover_only=False)
    providers = iter((interrupted_provider, recovery_provider))
    request_started = asyncio.Event()
    original_stream = causal_memory_campaign._CampaignScriptedProvider.stream

    async def interrupt_stream(
        provider: causal_memory_campaign._CampaignScriptedProvider,
        request: ModelRequest,
    ):
        provider.requests.append(ModelRequest.model_validate(request.model_dump(mode="python")))
        request_started.set()
        await asyncio.Future()
        async for event in original_stream(provider, request):
            yield event

    monkeypatch.setattr(
        causal_memory_campaign._CampaignScriptedProvider,
        "stream",
        interrupt_stream,
    )
    monkeypatch.setattr(
        causal_memory_campaign,
        "_scripted_provider",
        lambda *, recover_only: next(providers),
    )

    async def interrupt_campaign() -> None:
        task = asyncio.create_task(run_causal_memory_reference_campaign(corpus, state_directory))
        await asyncio.wait_for(request_started.wait(), timeout=30)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(interrupt_campaign())
    monkeypatch.setattr(
        causal_memory_campaign._CampaignScriptedProvider,
        "stream",
        original_stream,
    )
    report = asyncio.run(run_causal_memory_reference_campaign(corpus, state_directory))

    cancelled = [
        row
        for row in report.rows
        if row.execution_status is MemoryInterventionExecutionStatus.CANCELLED
    ]
    assert len(report.rows) == (
        len(corpus.cases)
        * CAUSAL_MEMORY_CAMPAIGN_REPETITIONS
        * len(CAUSAL_MEMORY_CAMPAIGN_VARIANTS)
    )
    assert len(cancelled) == 1
    assert cancelled[0].availability is MemoryTrialAvailability.CANCELLED
    assert cancelled[0].published_status == "unavailable"
    assert len(recovery_provider.requests) == len(report.rows) - len(interrupted_provider.requests)


def test_reference_campaign_rejects_a_different_standard_eval_corpus(tmp_path: Path) -> None:
    document = _CORPUS.read_text(encoding="utf-8").replace(
        "What was that rollback word?",
        "What was the earlier word?",
    )
    altered = tmp_path / "altered.json"
    altered.write_text(document, encoding="utf-8")

    with pytest.raises(ValueError):
        load_causal_memory_reference_corpus(altered)
