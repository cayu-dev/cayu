"""Selective successor admission with finite, deterministic per-slot allowances."""

from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path

from cayu.evals.benchmark_campaign import (
    BenchmarkCampaignSettingsV1,
    BenchmarkRetryLineageV1,
    admit_benchmark_campaign,
    load_benchmark_campaign,
    prepare_benchmark_campaign,
    resume_benchmark_campaign,
)
from cayu.evals.benchmark_inspection import inspect_benchmark_campaign
from cayu.evals.benchmark_package import load_benchmark_package
from cayu.evals.corpus import _content_revision
from cayu.evals.runner import EvalPlan
from cayu.storage.evals_sqlite import SQLiteEvalStore
from cayu.storage.migrations import SchemaMode


async def retry_benchmark_trial(
    directory: str | Path,
    plan: EvalPlan,
    *,
    case_id: str,
    trial_number: int,
    attempt: int = 1,
    allow_reexecution: bool = False,
) -> Path:
    """Never change an original result or silently choose a failed trial to rerun."""

    parent = load_benchmark_campaign(directory)
    if parent.retry_of is not None:
        raise ValueError("Select retries from the original campaign, not a successor.")
    if type(attempt) is not int or not 1 <= attempt <= parent.settings.max_retry_attempts:
        raise ValueError("Retry attempt exceeds the original campaign's reserved allowance.")
    inspection = await inspect_benchmark_campaign(directory)
    selected = next(
        (
            row
            for row in inspection.trials
            if row.case_id == case_id and row.trial_number == trial_number
        ),
        None,
    )
    if (
        selected is None
        or selected.result_state != "published"
        or selected.source_trial_revision is None
    ):
        raise ValueError("Retry requires an exact published original trial.")
    if selected.failure_category not in parent.settings.retry_categories:
        raise ValueError("Original trial is not eligible under the admitted retry categories.")
    policy = plan.execution_profile_policy
    safe_reset = (
        policy is not None
        and policy.reset_strategy == "application_managed"
        and policy.isolation_revision is not None
    )
    if not safe_reset and not allow_reexecution:
        raise ValueError("Retry requires an application reset contract or --allow-reexecution.")
    slot = _content_revision(
        {"campaign": parent.revision, "case": case_id, "trial": trial_number},
        "benchmark retry slot",
    )[7:]
    root = Path(directory).resolve() / "retries" / slot
    successor_directory = root / str(attempt)
    prior_directory, prior_run_id = Path(directory), selected.run_id
    if attempt > 1:
        previous = await inspect_benchmark_campaign(root / str(attempt - 1))
        if previous.status == "pending" or any(row.status == "passed" for row in previous.trials):
            raise ValueError("A preceding successor is pending or already passed.")
        prior_directory = root / str(attempt - 1)
        prior_run_id = previous.campaign.runs[0].spec.id
    prior_store = SQLiteEvalStore(
        prior_directory / "evals.sqlite3", read_only=True, schema_mode=SchemaMode.VALIDATE
    )
    try:
        prior_record = await prior_store.load_run(prior_run_id)
        if prior_record is None or prior_record.finished_at is None:
            raise ValueError("The preceding execution attempt has no terminal receipt.")
        eligible_at = prior_record.finished_at + timedelta(
            seconds=parent.settings.retry_backoff_seconds
        )
        if datetime.now(UTC) < eligible_at:
            raise ValueError(f"Retry backoff is active until {eligible_at.isoformat()}.")
    finally:
        await prior_store.close()
    lineage = BenchmarkRetryLineageV1(
        campaign_revision=parent.revision,
        run_id=selected.run_id,
        case_id=case_id,
        trial_number=trial_number,
        source_trial_revision=selected.source_trial_revision,
        attempt=attempt,
        failure_category=next(
            category
            for category in parent.settings.retry_categories
            if category == selected.failure_category
        ),
        replay_decision="caller_authorized" if allow_reexecution else "application_reset",
    )
    if not successor_directory.exists():
        package = load_benchmark_package(Path(directory) / "package")
        if package.package.revision != parent.package_revision:
            raise ValueError("Saved retry package does not match original admission.")
        settings = BenchmarkCampaignSettingsV1.model_validate(
            {
                **parent.settings.model_dump(mode="python"),
                "trials": 1,
                "minimum_passed_trials": 1,
                "max_concurrency": 1,
                "max_retry_attempts": 0,
            }
        )
        prepared = await prepare_benchmark_campaign(
            package,
            plan,
            settings=settings,
            case_ids=[case_id],
            campaign_id=f"retry-{slot}-{attempt}",
            retry_of=lineage,
        )
        # The model/environment/tools/reset/budget identity must still be the original one.
        original = next(run for run in parent.runs if run.spec.id == selected.run_id)
        original_profile = original.spec.invocation.execution_profile_snapshot
        successor_profile = prepared.campaign.runs[0].spec.invocation.execution_profile_snapshot
        if (
            original_profile is None
            or successor_profile is None
            or original_profile.comparison_revision != successor_profile.comparison_revision
        ):
            raise ValueError("Retry target does not match the original execution profile.")
        # Another caller may win the single deterministic admission slot.
        with suppress(FileExistsError):
            await admit_benchmark_campaign(prepared, successor_directory)
    successor = load_benchmark_campaign(successor_directory)
    if successor.retry_of != lineage:
        raise ValueError("Successor admission does not match the selected retry decision.")
    await resume_benchmark_campaign(successor_directory, plan)
    return successor_directory
