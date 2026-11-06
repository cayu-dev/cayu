# Runtime fault injection

These tests run real Runtime scenarios and inject one fault at every point a
scenario reaches, then check that durable state stays settled and recoverable.
They make exhaustive, for a fixed set of scenarios, the search for a bad
interleaving that is otherwise done by hand, one interleaving at a time.

```bash
# What pull requests run: the sample, the known-failure checks and the point check.
uv run pytest -q -m "fault_injection and not fault_injection_exhaustive" tests/fault_injection
# Every point.
uv run pytest -q -m fault_injection tests/fault_injection
```

The marker keeps them out of the general CI shards. The `Fault injection` CI job
runs on pull requests that touch `src/cayu/applications.py`, its
`src/cayu/_application_*` helpers, `src/cayu/approvals/`, `src/cayu/runtime/`,
`src/cayu/sessions/`, `src/cayu/tasks/`, `src/cayu/storage/` or this directory,
and on every push to `main`. Pull requests inject at the sample of points described below; pushes to
`main`, tags, manual runs and private release pull requests inject at every
point. Locally: `python3 scripts/run_ci.py --lane fault-injection`, with
`--exhaustive-fault-injection` for every point.

## How faults are injected

Every scenario uses a real `CayuApp` over `InMemorySessionStore` and
`InMemoryTaskStore`, a scripted provider and in-process tools
(`harness.py`). Three kinds of boundary report to one `Injector`:

- every top-level coroutine method of both stores, on entry, and for writes
  also on exit (after the write committed, before the caller sees the result).
  The stores are subclassed with plain-function overrides that redeclare the
  stores' capability flags, so the Runtime still uses their native protocols;
  `test_harness_store_parity.py` checks that every capability probe answers
  as it does for the plain stores;
- every scripted provider request and stream event;
- every tool call's start and end.

Only the scenario's phase under test is armed; its setup runs unarmed. A point
is a boundary plus its occurrence among equal boundaries, such as
`store.load:enter#3` or, for stream closes, `event:interaction.completed#0`.
Labels rather than positions keep points stable when an unrelated call moves:
an added or removed call shifts only the later occurrences of its own boundary.
Task lease renewals run on a wall-clock timer, so they are not numbered. A fault-free run lists the points; the exploration then
reruns the scenario in a fresh event loop for each point and fault:

| Fault | Injected at | What happens |
|---|---|---|
| `cancel` | every boundary | The task driving the phase is cancelled there. |
| `grouped-cancel` | every boundary | The task driving the phase is cancelled, and so is the task at the boundary if it is another one. The boundary then acts as cleanup code that receives that cancellation and fails too: it raises the real `CancelledError` grouped with an `OSError`, with the cancellation request still pending. The Runtime aggregates cleanup failures this way in many places. |
| `stream-close` | every event | The consumer closes the Runtime's event stream (`aclose()`, so `GeneratorExit` at the suspended `yield`) after that event. |
| `crash` | every store write | The process dies just before the write: it and every later store, provider and tool call of that app raise `SimulatedCrash`, so no `finally` block or background task can persist anything. Its leftover tasks are cancelled, and the store is handed to a new app. |

## Invariants

`invariants.py` checks, in the still-live process after the phase ends
(`live`), after `app.aclose()` (`after-shutdown`), and after a new app has run
`recover_abandoned_sessions()` with every lease expired (`recovered`):

- `session-running-without-owner`: no session is left `running` or
  `interrupting` once nothing executes it.
- `execution-lease-held-by-dead-owner`: no session reports an `executing` or
  `owner_lost` execution owner.
- `interaction-started-never-settled`: in a terminal session every started or
  resumed interaction has a terminal interaction event.
- `transcript-indexes-inconsistent` and `transcript-tool-result-without-call`:
  durable transcript indexes run `0..cursor-1`, and every tool result follows
  its call.
- `task-held-by-exited-worker`: no task stays `claimed` or `running` after its
  worker has returned or died, and the second worker settles it in recovery.
- `task-outcome-disagrees-with-session`: a task completes exactly when its
  linked session completes.
- `task-group-execution-obligation-unsettled`: every execution a quiescent task
  group recorded is settled once its member is terminal.
- `shutdown-or-recovery-hangs`: the phase, shutdown and recovery all finish,
  and `app.aclose()` reports a settled shutdown. A shutdown step that fails
  with the injected cleanup failure of a `grouped-cancel` fault is excused,
  since shutdown then reports exactly what was injected.
- `session-cannot-continue-after-recovery`: after recovery the session completes
  another turn. A pending approval is approved, or settled through manual
  recovery when the Runtime asks for it. A session whose setup never published
  its initial transcript is exempt: it fails closed by design.

Task scenarios recover with a second worker (`reconcile_abandoned_after_s=0`
and a `recovered_interrupted_task_handler` that resumes the task's session).
Recovery has finished when the task is terminal, or when that worker has polled
40 times in a row without claiming, running or changing it. Counting polls
rather than seconds keeps the check independent of the runner's speed.

Every stage of a run is bounded: the setup, the phase, each invariant check,
shutdown, recovery and the continuation turn. A stage that does not finish is a
`shutdown-or-recovery-hangs` violation of that point, and each run has its own
event loop, which never waits on a task that ignores cancellation. Should a run
hang outside an `await`, a watchdog prints its scenario, fault and point with
every thread's stack, then ends the process. Every point of the fault-free
trace must be reached again; a scenario whose trace changes between runs fails.

## Scenarios

| Scenario | Phase under test | Points per fault, sample / all (cancel; crash; close) |
|---|---|---|
| `run` | a plain text turn | 59/112; 18/35; 10/10 |
| `tool-round` | a turn with two parallel tool calls | 68/199; 20/61; 18/18 |
| `resume` | a resumed turn with a parallel tool round | 81/238; 19/66; 18/18 |
| `approval` | approving a reviewed call and finishing the turn | 95/197; 20/47; 14/14 |
| `task` | a worker claims a task and runs its session through a tool round | 101/246; 32/79; 20/20 |
| `task-group` | the same task as the member of a quiescent task group | 101/249; 32/80; 20/20 |

`grouped-cancel` uses the same points as `cancel`.

## Pull-request sample and run time

Most points repeat a boundary the Runtime already crossed from the same
function, such as the twentieth `mark_persisted_event_side_effect_delivered`.
The fault-free run records, for each boundary, the innermost Runtime function
that crossed it. The sample (`test_sampled_injection_points_keep_runtime_invariants`)
takes the first point of every distinct pair of boundary and function, plus
every stream-close point; `test_remaining_injection_points_keep_runtime_invariants`
(marked `fault_injection_exhaustive`) takes every other point. The two together
are the whole set, so `-m fault_injection` runs each point once.

The sample is 1,251 of the 2,950 runs, and it reaches at least one recorded
point of every entry in `known_failures.py`. Pull requests also run the strict
`xfail` check of every entry and the check that every recorded point is still
reached.

On one laptop core, the pull-request selection took about 20 minutes and every
point about 52 minutes; the slowest single test, `approval` under
`grouped-cancel` in the sample, took about 2.5 minutes. CI times are projected
from these, not measured: the job spreads the tests over four workers, so on a
runner with at least four cores the sample should take well under 10 minutes and
every point under 30. The job's bound is 40 minutes for the sample and 90 for
every point, which leaves room for a 2-vCPU runner with slower cores. Most of
each run is Runtime work: the setup turn of `resume` and `approval`, the phase,
and, after recovery, the turn that checks the session can continue, which is
between a half and two thirds of a session scenario's run.

## Failures found on `main`

`known_failures.py` records every violation found, by finding, scenario, fault
and point, with the invariants each point breaks and the stage each is found at
(in the live process, after shutdown, after recovery), and maps each finding to
the issue it reproduces or to its bug class. A known defect in the live process
therefore doesn't excuse a new one after recovery. The exhaustive tests accept exactly those and fail on anything new.
Each entry also has a strict `xfail` test that injects at its first point, so it
reports XPASS when a fix lands; then delete the entry. Points are cancellation
points unless the fault says otherwise.

| Finding | Faults, scenarios | Points | Invariant broken | Where |
|---|---|---|---|---|
| `task-completion-cancel` | grouped-cancel: task, task-group | 14 | task stays running under a completed session | between `_complete_session_if_no_queued_messages` and `terminalize_task` |
| `task-session-grouped-recovery` | grouped-cancel: task, task-group | 68 | session running, owner lost, even after recovery | a task-backed run's model steps and tool round; recovery then fails with `TaskClaimLost` because the task was already settled |
| `approval-grouped-continuation` | grouped-cancel: approval | 1 | session running without owner | right after the approval continuation commits `transition_status_and_checkpoint` |
| `grouped-model-stage` | grouped-cancel: run, tool-round, resume, approval | 12 | cannot continue after recovery ("no authoritative open interaction") | right after `complete_model_completion_stage` commits in `_assistant_model_publication` |
| `trailing-fence-cancel` | cancel, grouped-cancel: approval | 9 | completed session fenced forever (`SessionRunFenced`) | the approval continuation's trailing fence release in `cleanup_recovery_handoff` |
| `trailing-fence-crash` | crash: run, tool-round, resume, approval | 30 | completed session fenced forever; abandoned-session recovery skips completed sessions | between the interaction's completion and `release_session_invocation` |
| `task-claim-cancel` | cancel, grouped-cancel: task, task-group | 8 | task claimed by the exited worker until the next worker recovers it | while `mark_claimed_task_execution_started` is in flight; the claim waits for its pending execution-entry settlement |
| `task-crash-terminalize` | crash: task, task-group | 6 | task running forever under a completed session | between session completion and `terminalize_task` |
| `group-crash-conflict` | crash: task-group | 50 | task running forever: the next worker's continuation raises `TaskGroupConflict: Prior group execution is still unsettled` | a crash anywhere in the member's run |
| `group-crash-after-complete` | crash: task-group | 11 | group execution obligation never settled | a crash after the member completed, before `_settle_task_group_execution` |

Each finding names its tracking issue in `known_failures.py`; findings with no
issue of their own share one tracking issue there.

## What the scenarios do not reach

- Further scenarios (or setups) for this harness: a completed but unpromoted
  model stage before resume, a queued `on_idle` follow-up, model failover, a
  task-backed resume, manual recovery streams, runner cleanup artifacts,
  environment factories and differing execution profiles.
- Generation fencing needs two live workers racing for one lease, and
  durable-index shifts a model switch. The harness injects one fault into one
  owner; races between owners belong to protocol models.

## Adding a scenario

Add a `Scenario` to `SCENARIOS` in `explore.py`: an unarmed `setup` and a
`phase` that hands every Runtime event stream it opens to `consume`. Run the
new scenario's tests; record what fails on `main` in `known_failures.py` with
the issue it reproduces, or file one.
