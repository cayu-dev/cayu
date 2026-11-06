# Protocol models

Model-checked specifications of two Runtime recovery protocols. They are
written in [Quint](https://quint-lang.org) and checked exhaustively with
Quint's TLC backend over small instances. Each model has explicit cancel,
grouped-cancel, stream-close and crash actions.

| Model | File | What it covers |
|---|---|---|
| Session settlement | `session_settlement.qnt` | One run or resume of a session, optionally attached to a task. Admission, setup, body, the COMPLETED decision, `complete_session`, the trailing fence release, the worker's disposition, the abandoned-session sweep and the worker's expired-handoff scanner. |
| Task lease | `task_lease.qnt` | One task and two worker incarnations, possibly with the same `worker_id`. Claim, heartbeat with a late acknowledgement, expiry, reclaim, continuation claim, completion or failure with retries, interrupted-session handoff, stall and crash. |

Each model describes the code on `main` as variant `current` or `token`, and a
proposed fix as one or two more variants. Comments cite the code by function
name and file under `src/cayu/`. Each variant is a module instance at the end of
its file, and every check names one instance with `--main`.

## Running

Java 17 or later must be on `PATH`. Then:

```bash
npm ci --prefix formal/quint
npm run --prefix formal/quint check            # every check, one at a time
npm run --prefix formal/quint check cexHandoffReleasesSameNameReplacement    # only the named checks
```

The first run downloads the Apalache distribution, which contains TLC, into
`~/.quint`. One check runs:

```bash
cd formal/quint
JVM_ARGS=-Xmx1g npx quint verify --backend=tlc --tlc-config=tlc.json \
  --main=task_lease_token --invariant=cexHandoffReleasesSameNameReplacement task_lease.qnt
```

- `tlc.json` caps TLC at a 1 GB heap and two workers.
- `check.mjs` gives the Quint-to-TLA+ compiler the same 1 GB cap, runs the
  checks one at a time, and stops any check after three minutes.
- Add `--verbosity=3` to print TLC's output, including the counterexample trace.
- For quick exploration without TLC, run
  `npx quint run --main=<instance> --invariant=<name> <file>` to simulate random
  traces.

The suite has 36 checks and takes 4 to 8 minutes on a laptop, depending on
load. Most of that is JVM start-up, 6 to 9 s per check. The two exhaustive task
lease checks take 15 to 40 s each, over 1.2 to 1.3 million distinct states. The
session model has about a thousand states.

The `Protocol models` workflow (`.github/workflows/protocol-models.yml`) runs
the suite on pull requests that touch `formal/quint/`, `src/cayu/applications.py`
or `src/cayu/{runtime,sessions,storage,tasks}/`. Its check step is bounded at
15 minutes, and it caches `~/.quint` by the lockfile.

## How checks work

Every property is a state invariant, so TLC reports a shortest counterexample.
There are four kinds of check:

- **Expected to hold:** a property of a fixed variant. TLC must find no
  violation.
- **Violated on `main`** (`current...`, `token...`): each settlement or lease
  property, limited to today's code. These back the claim that today's code
  breaks every property.
- **Known counterexample** (`cex...`): the negation of one bad state, limited to
  one exit or one configuration, so TLC reports exactly that trace. It is
  expected to be **violated**.
- **Reachability** (`reach...`): shows the good path is reachable, so a property
  that holds is not vacuous. It is expected to be **violated**.

The checks run only the models. They never import or execute `src/cayu`, so a
Runtime fix leaves every known counterexample violated and CI green. The model
has to be updated by hand to match a fix: change the `current` handler table or
the `token` checks. The counterexample then reports "no violation", `check.mjs`
fails, and its expectation moves to "holds" in `check.mjs` and in this file.

"Recovery has finished" is written as a safety condition, `quiescent`: the run
has returned or crashed, and no recovery action is enabled. A property such as
"no session is RUNNING after recovery" then reads
`quiescent implies status != RUNNING`.

## Session settlement model

**Locations.** These are the run's locations on `main`, with the handler that
owns each one. All are in `runtime/_session_engine.py` unless stated.

| Location | Code | close | cancel / grouped cancel | raise |
|---|---|---|---|---|
| `run.admitted` | `SessionEngine.run`, after the `create` lifecycle command commits and before the setup `try` | – | escapes | escapes |
| `run.setup` | `SessionEngine.run`'s setup `try`, starting with `execution_presence.ensure` | – | abandon | fail |
| `run.started` | `SessionEngine.run`'s INTERACTION_STARTED yield | interrupt | – | – |
| `res.admitted` | `SessionEngine.resume`, after its admission command and before the setup `try` | – | escapes | escapes |
| `res.setup` | `SessionEngine.resume`'s setup `try` | abandon | abandon / **escapes**: a group reaches `except BaseException`, which only tries the fence release | fail |
| `res.promoted` | `SessionEngine.resume` yields the recovered MODEL_COMPLETED before its inner stream starts | **escapes**: only the unstarted inner generator is closed | – | – |
| `body` | `SessionEngine._run_session_with_deadline`'s protected `try` | interrupt | abandon | fail |
| `decided` | after `_complete_session_if_no_queued_messages`, before the INTERACTION_COMPLETED yield | **escapes** (settled-interaction guard) | **escapes** (settled-interaction guard) | – |
| `completing` | `SessionOutcomePublication.complete_session`, before it completes the task | – | **escapes** | – |
| `fence` | the trailing release, `EnvironmentLifecycle._release_quiescent_invocation_fence` (`runtime/_environment_lifecycle.py`) | – | **escapes** | – |

"Escapes" means nothing durable is written. The settled-interaction guard is the
`_latest_session_invocation_interaction_is_settled` check in the body's
`CancelledError` and `GeneratorExit` handlers.

**Recovery:**

- **The abandoned-session sweep.** `SessionAbandonment.recover_abandoned`
  (`runtime/_session_abandonment.py`) runs `recover_incomplete_sessions` over
  RUNNING and INTERRUPTING sessions. `recover_incomplete_session_scoped` skips
  only a live execution owner. The sweep runs only when the application
  schedules `CayuApp.recover_abandoned_sessions`, so the model makes it a
  configuration choice (`cfg.sweep`).
- **The worker's expired-handoff scanner**
  (`_recover_expired_interrupted_task_handoffs`, `tasks/worker.py`). It recovers
  a RUNNING session, then hands the task off only when the session is
  INTERRUPTED.
- **`InMemoryTaskStore.reclaim_expired`**, for a claim whose session was never
  created.

**Worker disposition** (`current`):

- `_await_task_handler` re-raises a grouped cancellation for an ordinary task.
  `_handle_with_heartbeat_owned` catches only `Exception`, so no disposition
  runs.
- Otherwise an INTERRUPTED session is handed off
  (`_handoff_interrupted_session`).
- Anything else still running is failed (`_safe_fail`).

**Variants:**

- `current`: the table above.
- `design`: a single settlement path. One wrapper owns admission and the
  consumer-facing stream, and every exit of every location goes to one
  idempotent `settle`.
  - `settle` runs decide, then the task, then the terminal status, then the
    fence, one durable step at a time. A crash can come between any two steps.
  - The FAILED or INTERRUPTED terminal decision is written first, with the
    session still RUNNING, as `claim_invocation_terminal_decision` and
    `ensure_interruption_terminal_decision` already do.
  - The COMPLETED decision is written in the same transaction as the COMPLETED
    transition.
  - Recovery is the existing sweep over RUNNING sessions and the worker's
    expired-handoff scanner, both ending in the same decision-aware `settle`:
    a recorded FAILED decision fails the task instead of handing it off. The
    design also reconciles orphaned sessions itself, so the sweep is always
    on.
- `design_scan`: `design` plus a recovery scan of COMPLETED sessions that still
  hold an unsettled decision or an unreleased fence.

**Properties.** `settled` is the conjunction of these:

| Property | Meaning |
|---|---|
| `noOrphanedRunning` | After recovery, no session is RUNNING. |
| `runExitSettles` | Once the run has returned and before any recovery, the session is not RUNNING. |
| `interactionSettles` | After recovery, every started interaction has settled. |
| `taskMatchesSession` | After recovery, the task is not running, and it is COMPLETED exactly when the session is. |
| `workerReleasesTask` | A worker that has returned holds no task. |
| `noPermanentFence` | After recovery, no COMPLETED session keeps its run fence. |

**Results:**

- `design_scan` satisfies `settled` (check `designScanSettles`).
- `design` satisfies it in every trace except a crash after the COMPLETED
  decision (check `designSettlesWithoutCrashAfterDecision`).
- `current` violates each of the six properties (checks
  `currentNoOrphanedRunning` to `currentNoPermanentFence`).

## Task lease model

The task row has `status`, `owner` (`worker_id`), `tok` (the lease token, which
is the exact `lease_expires_at`), `live` and `gen`. Each incarnation keeps three
things:

- the token it believes it holds;
- a renewal whose acknowledgement is still outstanding;
- the generation it claimed.

A lease lapses only while its holder's heartbeat loop is not running: the loop
was stopped or stalled, or the process crashed. Under that assumption a
heartbeat is meant to keep a lease alive. A `stall` is a genuine loss of
authority, and the properties exempt it.

**Variants:**

- `token` (`main`):
  - Every claimed mutation and heartbeat checks `owner == worker_id`, a live
    lease and the exact token (`_ensure_exact_owned_active_task_lease`,
    `tasks/_lifecycle.py`; `InMemoryTaskStore.heartbeat`).
  - Every heartbeat mints a new token.
  - Ordinary tasks stop the heartbeat before the disposition
    (`stop_heartbeat.set()` in `_await_task_handler`). The disposition starts
    only after the heartbeat task has ended, so any pending acknowledgement is
    applied first, and it runs under `lease_authority.lock`.
  - Retry-series tasks settle with the latest acknowledged token and retry up
    to three times (`_settle_task_retry_request_with_lease_authority`).
  - The handoff preflight (`_handoff_interrupted_session`) builds its request
    from the row it just read.
- `generation`: a proposed generation fence. Claim, reclaim, handoff and
  continuation claim increment `gen`, and heartbeats keep it. Every mutation
  checks `gen == the callback's generation` and a live lease.
- `generation_hb_last`: `generation` plus one ordering rule. The heartbeat stops
  only after the disposition has committed.

**Properties.** `safe` is the conjunction of these:

| Property | Meaning |
|---|---|
| `noStaleOwnerWrite` | Only the holder of the current claim changes the task. |
| `noFalseClaimLost` | The store never refuses the holder while its lease is live. |
| `reportRecorded` | A callback that kept its heartbeat has its outcome recorded when its worker returns. |
| `liveHolderKeepsClaim` | Nobody else releases or terminalizes a running callback's claim. |

`token` violates each of the four (checks `tokenNoStaleOwnerWrite` to
`tokenLiveHolderKeepsClaim`). `tokenNoFalseClaimLost` is the mechanism under
`cexRetrySettlementLosesToHeartbeats` and under a late heartbeat
acknowledgement in the dispatcher: a write sent while a renewal's
acknowledgement is outstanding carries the previous token, and the store
refuses the live holder. `tests/core/test_dispatch_late_renewal_settlement.py`
covers the dispatcher case.

**What the generation-fenced variant proves.** The instance covers:

- two incarnations, with the same or different worker names;
- attached and unattached tasks, ordinary and retry-series;
- every interleaving of claim, heartbeat, late acknowledgement, expiry,
  reclaim, handoff, settlement retries, stall and crash;
- up to 7 lease tokens.

In that instance:

- **`generation_hb_last`** satisfies all four properties
  (`generationHbLastSafe`). This is the spec for a generation-fencing fix:
  - fence every claimed mutation by a generation that heartbeats keep and that
    reclaim and handoff change;
  - bind the disposition to the callback's generation;
  - stop the heartbeat last.
- **`generation` alone** satisfies `noStaleOwnerWrite`, `noFalseClaimLost` and
  `liveHolderKeepsClaim` (`generationFences`). This removes
  `cexHandoffReleasesSameNameReplacement`,
  `cexRetrySettlementLosesToHeartbeats` and the stale-acknowledgement refusal.
  It does **not** satisfy `reportRecorded`:
  - `cexHeartbeatStoppedBeforeDisposition` is a heartbeat-ordering bug, not an
    identity bug;
  - the generation check still requires a live lease
    (`cexGenerationNeedsHbLast`).

## Known counterexamples

Each trace is TLC's shortest. A dash in the regression column means no test on
`main` reproduces it yet.

### Session settlement (`current`)

| Check | What goes wrong | Trace | Regression |
|---|---|---|---|
| `cexResumeSetupGroupCancel` | Grouped cancellation in resume setup leaves the session RUNNING. | Resume admits (RUNNING, fence open) → setup → grouped cancellation → `except BaseException` tries only the fence release, which is refused → the run returns with the session RUNNING. | – |
| `cexResumeAdmissionCancel` | Cancellation as resume admission commits leaves the session RUNNING. | Resume admission commits → cancellation lands on the admission's await → nothing owns the admitted run before the setup `try` → the session stays RUNNING. | – |
| `cexRunEntranceCreate` | Cancellation right after `create` leaves the session RUNNING with no owner. | `create` commits RUNNING → cancellation before the setup `try` → the session stays RUNNING with no execution owner. | – |
| `cexResumePromotedClose` | Closing at a recovered model completion strands the resumed session. | Resume admits → setup promotes the retained stage → the consumer closes at the promoted MODEL_COMPLETED → only the unstarted inner generator is closed, and the outer `finally` only deactivates → the session stays RUNNING with owner `unknown`. | – |
| `cexCompletedInteractionClose` | Closing at INTERACTION_COMPLETED skips the task's completion. | The body commits the COMPLETED decision → close at INTERACTION_COMPLETED → the settled-interaction guard re-raises, so `complete_session` never runs → the worker fails the task of the completed session. With the task attached directly and no worker, the task stays RUNNING. | – |
| `cexTaskCompletionGroup` | Grouped cancellation after the decision leaves the task running forever. | COMPLETED decision → grouped cancellation before the task is completed → the guard re-raises → the worker re-raises the group → the task stays running under the completed session. The scanner skips COMPLETED sessions. | – |
| `cexGroupCancelHoldsTask` | Grouped cancellation leaves the task held by the exited worker. | Body → grouped cancellation → the session is settled INTERRUPTED → the worker re-raises the group without a disposition → the returned worker still holds the task. | – |
| `cexTaskCrashTerminalize` | A crash after the decision leaves the task running forever. | COMPLETED decision → crash before the task is completed → the task stays running forever: the scanner skips COMPLETED sessions, and the sweep visits only RUNNING ones. | – |
| `cexTrailingFenceCancel` | Cancellation at the trailing release fences the completed session forever. | Decision → task completed → cancellation at the trailing release → the completed session keeps its fence, and every later run is refused. | – |
| `cexTrailingFenceCrash` | A crash before the trailing release fences the completed session forever. | Decision → task completed → crash before the trailing release → the fence stays; recovery skips COMPLETED sessions. | – |
| `cexOrphanedRunningNoSweep` | A session orphaned by a crash is never reconciled. | A run with no task is admitted → crash → the session stays RUNNING forever when nothing schedules `recover_abandoned_sessions`. | – (sketch 1) |
| `cexOrphanedRunningTaskFailed` | The same with a task. This is a model prediction, not reproduced yet. | Task-backed run admitted → cancellation before the setup `try` → `_safe_fail` fails the task, because the session is not INTERRUPTED → the scanner skips the session, because its task is terminal → RUNNING forever without the sweep. | – (sketch 2) |

On `main`, the opt-in sweep hides most of the RUNNING counterexamples after
recovery, when it is scheduled:

- They still break `runExitSettles`.
- `cexTaskCrashTerminalize`, `cexTrailingFence*` and `cexCompletedInteractionClose` survive recovery,
  because no recovery path visits a COMPLETED session.
- `EnvironmentLifecycle.release_settled_invocation` releases
  only settled pauses and interruptions. It does not release COMPLETED
  sessions, so these counterexamples still describe `main`.

### Session settlement (`design`)

**`cexDesignNeedsScan`.** In the `design` variant, recovery reaches `settle`
only through the sweep, which visits RUNNING and INTERRUPTING sessions. The
COMPLETED decision is written together with the COMPLETED status. So a crash
between that write and the end of `settle` is never finished. Any fix with this
shape needs a recovery scan for unsettled COMPLETED decisions; `design_scan`
adds it.

- **Trace:** resume admits → the body commits COMPLETED with its decision →
  crash → the fence stays open, and an attached task stays running.
- **Regression:** sketch 3.

### Task lease (`token`)

| Check | What goes wrong | Trace | Regression |
|---|---|---|---|
| `cexHandoffReleasesSameNameReplacement` | A delayed handoff releases a live replacement with the same worker ID. | Incarnation 1 (`w`) claims an attached task (token 1) → its handler returns a handoff, and the heartbeat stops → the lease expires → the scanner releases the claim → incarnation 2, also `w`, claims it (token 2) → incarnation 1's preflight reads the row (owner `w`, token 2) and commits the handoff → incarnation 2's claim is released while its callback runs. | – |
| `cexRetrySettlementLosesToHeartbeats` | Retry settlement exhausts its retries against its own heartbeats. | Retry-series task: claim (token 1) → a heartbeat commits token 2, acknowledgement pending → the settlement sends token 1 → the acknowledgement arrives → refused → retry with token 2 → another heartbeat commits token 3 → refused → retry with token 3 → heartbeat to token 4 → refused → three attempts are spent, and `TaskClaimLost` is raised with the lease still live. | – |
| `cexHeartbeatStoppedBeforeDisposition` | Stopping the heartbeat before the failure disposition abandons the task. | Ordinary unattached task: the handler returns, and the heartbeat stops and is joined → the failure is sent → the lease expires → refused for the expiry → `_safe_fail_payload` suppresses `TaskClaimLost` → the worker returns with the task CLAIMED and its outcome unrecorded. The check counts only a refusal caused by expiry. | – |
| `cexGenerationNeedsHbLast` | Generation fencing alone does not fix `cexHeartbeatStoppedBeforeDisposition`. | The same trace under `generation`, on the handoff path: the read, then expiry, then a refused commit. | – (sketch 4) |

A fault injected into one owner cannot reach the lease counterexamples: each
one needs two owners racing, or a late acknowledgement.

### Regression test sketches

These are sketches only. Each belongs with the change that fixes its
counterexample.

1. **`cexOrphanedRunningNoSweep`.**
   - Run a session on a `CayuApp` over a SQLite store.
   - Stop the process right after admission, for example by copying the store
     file.
   - Open a new app on the store, expire every lease, and start a task worker
     with default settings.
   - After one maintenance cycle, assert the session is not RUNNING, or that a
     recovery item exists.

   Today it stays RUNNING until the application calls
   `recover_abandoned_sessions`.
2. **`cexOrphanedRunningTaskFailed`.**
   - A worker handler calls `app.run` for an attached task.
   - A store wrapper raises `CancelledError` on the first `create`
     acknowledgement.
   - Assert that after the worker returns and one more maintenance cycle, the
     session is INTERRUPTED and the task is handed off or failed consistently.

   The model predicts task FAILED with the session RUNNING.
3. **`cexDesignNeedsScan`.**
   - With `tests/core/_session_operation_fault_harness.py`, inject
     `CommitThenRaise` on the first write after the COMPLETED transition: the
     task completion or the terminal event.
   - Recover with `recover_abandoned_sessions(inactive_for_seconds=0)`.
   - Assert the task is COMPLETED and the session's fence is released: a new
     run on the session is admitted.
4. **`cexGenerationNeedsHbLast`.**
   - Use a one-second lease and a store whose `load_task` sleeps 1.2 s after
     the handler raises.
   - Run the failure disposition against a generation-fenced store, with the
     heartbeat still stopped before the disposition.
   - The failure is refused. It is recorded only once the heartbeat stops last.

## Gaps

- **No link to the code.** The models abstract the code. Each location and
  handler is checked by reading it, not by refinement against the
  implementation, so a change in `src/cayu` does not fail these checks. Its
  author updates the model, which is why the workflow runs on those paths.
- **Not in the session model:**
  - queued successors;
  - manual-recovery entrances;
  - paused outcomes;
  - workspace `pending_completion_finalization`;
  - task-group execution obligations;
  - cancellation while the public API projects an event;
  - grouped cancellation right after a model completion stage commits;
  - the continuation's own takeover of an `owner_lost` session
    (`recover_abandoned_execution`).
- **Assumed, not checked:**
  - An INTERRUPTED or FAILED session with an unreleased fence is assumed to be
    released by the next admission or recovery.
  - The worker's disposition after an abnormal run exit is reduced to "hand off
    if INTERRUPTED, else fail". The worker's cancellation drain in
    `_await_task_handler` can instead let the run finish naturally.
- **`cexOrphanedRunningTaskFailed` is a model prediction.** It has no reproduction yet.
- **Task model limits:**
  - It has one task and two incarnations.
  - Handoff settlement does not retry, although
    `_settle_interrupted_task_handoff_with_retry` does in the code.
  - `cexRetrySettlementLosesToHeartbeats` is checked only for unattached tasks.
  - Cancellation requests, `restart_task` and the dispatcher's own fence are
    left out.
- **Not modeled yet:** worker maintenance ownership, and the Cloud
  publication-lock and import, export and archive protocols.
