# Measuring CI feedback time

Use `scripts/report_ci_timings.py` to compare completed GitHub Actions runs with
the same validation profile. It uses Python's standard library and authenticated
GitHub CLI (`gh`) GET requests. It does not start, retry, cancel or modify workflows.

```bash
python3 scripts/report_ci_timings.py \
  --repo OWNER/REPOSITORY --run-id RUN_ID \
  --snapshot /tmp/ci-snapshot.json > /tmp/ci-timing.json

# Reproduce the same report without GitHub access.
python3 scripts/report_ci_timings.py \
  --input /tmp/ci-snapshot.json > /tmp/ci-timing-replayed.json
```

The snapshot preserves API evidence for every attempt and every page of jobs.
Keep run-specific snapshots outside the source tree: they include the API's runner
metadata. The report selects timing fields and omits runner names, IDs and labels.
Compare private and public runner pools separately, recording the commit, workflow
profile, concurrent load and resource configuration alongside the report.

## Metric definitions

- `elapsed_to_selected_completion_seconds`: workflow creation to the last
  non-skipped selected job's completion in the final attempt. This includes earlier
  attempts and the delay before a retry. Workflow creation is the API's observable
  start; any event-delivery delay before creation is unavailable.
- `successful_selected_elapsed_seconds`: the same interval, populated only when
  all selected jobs finish successfully or are skipped, with at least one success.
  Failed, cancelled and entirely skipped selections do not establish time to green.
- `run_to_created_seconds`: workflow creation to job creation. This can include
  dependencies and retry delays; it is not a runner queue measurement.
- `created_to_started_seconds`: job creation to job start. This is observed job
  waiting, which may include scheduling, concurrency or approval delays. It does
  not establish how much time was caused specifically by insufficient runners.
- `execution_seconds` and `steps[].seconds`: job and step wall time. The report
  retains step names and outcomes so setup, collection, tests, packaging and
  cleanup can be inspected at the granularity actually exposed by the workflow.
- `observed_selected_job_execution_seconds`: total selected job execution across
  all attempts. Parallel jobs add together, so this differs from elapsed time.
  Successful jobs carried into a retry retain their original execution reference
  and contribute only once. This is not CPU time or billed runner cost.
- `observed_selected_job_waiting_seconds`: the corresponding sum of observed job
  waiting, excluding carried executions and skips. Missing timestamps are listed
  explicitly in `jobs_with_unavailable_execution_seconds` and
  `jobs_with_unavailable_waiting_seconds`; neither aggregate invents durations for
  missing observations or reversed intervals.
- `gap_after_previous_selected_completion_seconds`: the next attempt's start
  minus the prior attempt's selected completion. A negative value indicates an
  overlap. The gap can include excluded work and the wait before requesting a retry.

Missing timing observations remain `null`. GitHub can also report reversed
timestamps, including cancelled jobs that appear to finish before starting. Those
intervals become `null`, with the affected metric named in the job's
`timing_anomalies` list (step metrics use `steps.<number>.seconds`). Original
timestamps and conclusions are preserved. An anomalous job completion makes the
selected completion time unknown if that job is selected and not skipped; a
reversed step interval leaves valid job timing usable. Excluded jobs retain their
diagnostics without affecting selected measurements. Reversed overall elapsed
time is listed in the report's own `timing_anomalies`.

Incomplete pagination, inconsistent attempt identities, unfinished runs and a
rerun started during retrieval cause an error.
Reports include historical failures even when a retry succeeds. The final
selection's conclusion counts retain existing skips for review.

## Validation scope and publishing

By default every workflow job is selected. For release validation, explicitly
exclude publication jobs so their approval and publishing time do not extend the
measurement:

```bash
python3 scripts/report_ci_timings.py \
  --input /tmp/ci-snapshot.json \
  --exclude-job 'Publish to PyPI' \
  --exclude-job 'Create GitHub release' > /tmp/ci-validation-timing.json
```

Exclusions match exact names, are recorded in the report and must exist in the
snapshot. Do not exclude required validation checks. The reporter does not read
branch protection, prove that skipped checks were justified, or infer whether
full qualification was required. Verify that the workflow contains the intended
checks before interpreting its selected results as complete validation. Measure
separate qualification workflows independently; their success is not automatically
part of an ordinary CI run's result.

## What the API cannot measure

The current test step includes dependency sync, pytest startup, worker collection
and execution. Step timing alone cannot separate those costs. CPU allocation,
utilization, memory pressure and I/O also require runner-side measurements.
Record those separately before choosing worker counts or provisioning capacity.
Shorter elapsed time is useful only when the required inventory, assertions,
backend coverage and artifact checks remain intact.

API reference: [workflow runs](https://docs.github.com/en/rest/actions/workflow-runs)
and [workflow jobs](https://docs.github.com/en/rest/actions/workflow-jobs).
