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

## Profile test execution

Use the same lane runner with optional process measurements:

```bash
python3 scripts/run_ci.py --lane general --shard 1 \
  --test-workers 4 --profile-tests /tmp/ci-profile
```

The report is written to `/tmp/ci-profile/general-1.json` and printed as one
`CI execution profile:` JSON line. Profiling is off by default and can also be
used independently with `pytest -p scripts.ci_test_profile --ci-profile PATH`.
For independent invocations with output outside the repository, pass
`--rootdir . -c pyproject.toml` so the output path cannot alter pytest discovery.
The complete local CI runner accepts these options too; its parallel shards
write separate reports to the requested directory.

`--test-workers` accepts 1–8 workers for general and parallel specialist lanes.
The default is four. Qualification and host-journey lanes always retain one
worker because their fixtures create their own concurrency. Work-stealing
distribution, shard plan assignments and required backend flags are unchanged.

The report distinguishes these measurements:

- Each worker's collection and execution wall time, CPU user/system time,
  major page faults, block I/O and context switches.
- Resources charged to the worker itself versus terminated children it has
  waited for. External Docker services and still-running descendants are not
  included in these CPU totals. Linux process byte-I/O counters include waited
  children and are labelled separately.
- Process-lifetime peak resident memory, in bytes. Child peak memory is the
  operating system's child high-water measurement; it is not concurrent memory
  across a process tree. Peaks are neither phase deltas nor additive across
  workers. The controller's child counters can overlap worker measurements.
- Shared-host observations at session start/end: CPU count/affinity, load
  average, available memory, swap and Linux pressure-stall totals. Visible
  cgroup-v2 limits are recorded from the current group through its mounted
  ancestors; limits outside a container's visible hierarchy remain unknown.
  Cgroup memory and throttling counters can include other processes in that
  group. These observations do not isolate this job's contribution.
- Collection/execution inventory hashes, test outcomes and the twenty slowest
  tests/files by accumulated setup/call/teardown time. Summed test durations can
  overlap in parallel and are not elapsed CI time.

Linux-only or unreadable metrics are `null`; they are not reported as zero.
Reset counters also produce `null` deltas. Collection and test failures still
fail pytest. A crashed worker has an incomplete profile rather than fabricated
zero usage. Reports omit environment variables, command lines, runner identities
and filesystem locations; keep run-specific results outside the source tree.
Process measurements begin at pytest session start, so interpreter/plugin startup
and final process exit remain part of the outer command timing.

### Compare worker counts

The manual **CI execution profiling** workflow runs general shard 1 and PostgreSQL
shard 1 with 2, 4 and 8 workers, using an exact commit SHA and the pull request
shard plan. Each lane, worker count and repetition gets its own job: six jobs for
one repetition, twelve for the default two. At most two jobs run concurrently.
Use one repetition for a preliminary observation and both before drawing a
performance conclusion.

Each job uploads its JSON/log artifact immediately after its run and retains it
for seven days. Artifact names include the lane, worker count, repetition and run
attempt, so a retry preserves earlier results. Test failures remain visible and
do not cancel other comparisons. The test step is limited to 45 minutes within a
55-minute job, leaving time for setup and a bounded artifact upload after a test
timeout. A disconnected runner can still lose its current measurement; results
already uploaded by other jobs remain available. Ordinary PR/release CI does not
invoke this experiment.

Compare successful runs with identical collection and executed-inventory hashes,
outcomes and fixture ownership. Jobs can use different VMs: match actual CPU and
memory capacity, record simultaneous workflow load and inspect the profiles'
load, pressure and cgroup limits before comparing times. Independent repetitions
do not control VM variation or cache warmth. The public and private pools may have
different capacity; eight workers can oversubscribe a small runner. A measurement
on one shard or runner does not establish a new default or the full validation
target.

Measurement semantics: [Python resource usage](https://docs.python.org/3/library/resource.html),
[Linux process I/O](https://man7.org/linux/man-pages/man5/proc_pid_io.5.html),
and [cgroup v2](https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html).

API reference: [workflow runs](https://docs.github.com/en/rest/actions/workflow-runs)
and [workflow jobs](https://docs.github.com/en/rest/actions/workflow-jobs).
