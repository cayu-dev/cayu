"""Fail-closed campaign comparability and score deltas over exact native trials."""

from math import isfinite

from cayu.evals.benchmark_inspection import inspect_benchmark_campaign


def _identity(campaign):
    profiles = tuple(
        sorted(
            (
                case_id,
                run.spec.invocation.execution_profile_snapshot.comparison_revision,
            )
            for run in campaign.runs
            for case_id in run.case_ids
        )
    )
    return {
        "package": (campaign.package_id, campaign.package_version, campaign.package_revision),
        "cohort": campaign.selection.revision,
        "scorer": (campaign.scorer_id, campaign.scorer_version),
        "execution_profiles": profiles,
        "budgets_and_retry_policy": campaign.settings.model_dump(mode="json"),
        "retry_lineage": None
        if campaign.retry_of is None
        else campaign.retry_of.model_dump(mode="json"),
    }


async def compare_benchmark_campaigns(baseline, current, *, score_tolerance=0.0):
    if not isfinite(score_tolerance) or score_tolerance < 0:
        raise ValueError("Score tolerance must be finite and nonnegative.")
    left = await inspect_benchmark_campaign(baseline)
    right = await inspect_benchmark_campaign(current)
    left_identity, right_identity = _identity(left.campaign), _identity(right.campaign)
    mismatches = [key for key in left_identity if left_identity[key] != right_identity[key]]
    limitations = []
    if any(
        row.result_state != "published" or row.score is None
        for row in (*left.trials, *right.trials)
    ):
        limitations.append("unscored_or_unpublished_trials")
    comparable = not mismatches and not limitations
    deltas = []
    if comparable:
        baseline_rows = {(row.case_id, row.trial_number): row for row in left.trials}
        for row in right.trials:
            original = baseline_rows[(row.case_id, row.trial_number)]
            assert row.score is not None and original.score is not None
            delta = row.score - original.score
            deltas.append(
                {
                    "case_id": row.case_id,
                    "trial_number": row.trial_number,
                    "baseline_trial_revision": original.source_trial_revision,
                    "current_trial_revision": row.source_trial_revision,
                    "score_delta": delta,
                    "regressed": delta < -score_tolerance,
                }
            )
    return {
        "schema_version": 1,
        "baseline_revision": left.campaign.revision,
        "current_revision": right.campaign.revision,
        "compatibility": "incompatible"
        if mismatches
        else "unknown"
        if limitations
        else "comparable",
        "mismatches": mismatches,
        "limitations": limitations,
        "trials": deltas,
        "regressions": sum(row["regressed"] for row in deltas),
    }
