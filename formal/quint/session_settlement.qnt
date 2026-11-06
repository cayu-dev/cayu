// -*- mode: Bluespec; -*-
/**
 * Session settlement across every exit path of one admitted invocation.
 *
 * One session, optionally attached to one task, is run (`SessionEngine.run`) or
 * resumed (`SessionEngine.resume`) by one process. At every await the run can be
 * cancelled (plain `CancelledError`) or receive a grouped cancellation
 * (`BaseExceptionGroup` holding a `CancelledError`), at every yield the consumer
 * can close the stream (`aclose()`, so `GeneratorExit`), setup and body code can
 * raise, and the process can crash between any two store writes. Recovery is
 * the abandoned-session sweep and the task worker's expired-handoff scanner.
 *
 * Three variants share the state machine:
 *
 *  - "current": the exit handlers as they are on `main`. Each location has its
 *    own hand-written handler, so the model maps (location, exit) to what that
 *    handler does.
 *  - "design": a single settlement path: one wrapper owns admission and the
 *    consumer-facing stream, and every exit of every location goes through one
 *    idempotent `settle(exit)` that decides the outcome, settles the task,
 *    publishes the terminal status, then releases the fence. A crash leaves the
 *    open invocation or a durable terminal decision, and recovery runs the same
 *    steps. Recovery is the existing sweep over RUNNING and INTERRUPTING
 *    sessions.
 *  - "design_scan": "design" plus a recovery scan that also visits COMPLETED
 *    sessions holding an unsettled decision or an unreleased fence.
 *
 * Citations name functions under `src/cayu/`.
 */
module session_settlement {
  /// The variants a check explores; the instances at the end of this file fix
  /// one each.
  const VARIANTS: Set[str]

  // ---------------------------------------------------------------------------
  // State
  // ---------------------------------------------------------------------------

  /// Configuration, chosen once in `init` and never changed.
  ///  variant:  "current" | "design" | "design_scan"
  ///  hasTask:  the session is attached to a task run by a worker in this process
  ///  sweep:    something calls `CayuApp.recover_abandoned_sessions`
  ///            (applications.py; the application must schedule it, and nothing
  ///            in the task worker calls it)
  ///  promoted: resume retains a completed but unpromoted model stage
  var cfg: { variant: str, hasTask: bool, sweep: bool, promoted: bool }

  type Db = {
    status: str, fenceOpen: bool, inter: str, task: str, decision: str
  }

  /// Durable state in the session store and the task store.
  ///  status:     session status: "NONE" | "RUNNING" | "COMPLETED" | "INTERRUPTED" | "FAILED"
  ///  fenceOpen:  an unreleased active profile at the current run epoch;
  ///              admission installs it and release clears it
  ///              (`AdmitInvocationCommand`, `ReleaseInvocationCommand`,
  ///              sessions/_invocation_lifecycle.py;
  ///              `active_invocation_execution_profile_from_checkpoint`,
  ///              sessions/_execution_profile_checkpoint.py). Epoch numbers
  ///              and the execution-owner row are left out: no recovery
  ///              decision here depends on them, only on whether the run is
  ///              still active in its process.
  ///  inter:      the latest interaction: "none" | "started" | "completed" |
  ///              "interrupted" | "failed"
  ///  task:       attached task: "none" | "running" | "completed" | "failed" |
  ///              "handoff" (released for a continuation worker) | "pending"
  ///              (reclaimed before its session existed)
  ///  decision:   design only: the durable InvocationTerminalDecision,
  ///              "" | "completed" | "failed" | "interrupted"
  ///              (sessions/_invocation_terminal_decision.py; on `main` it
  ///              exists for FAILED and INTERRUPTED, the design adds COMPLETED)
  var db: Db

  /// The process.
  ///  alive:     false after a crash
  ///  pc:        where the run is (see LOCATIONS); "exited" = the run's stream
  ///             ended and the worker has not finished its disposition yet;
  ///             "settling" = inside the design's settle(); "returned" = the run
  ///             and its worker are done; "dead" = crashed
  ///  groupExit: the run ended with a grouped cancellation
  ///  failExit:  the run ended with an ordinary exception
  var proc: { alive: bool, pc: str, groupExit: bool, failExit: bool }

  /// History, for the properties and for readable traces.
  ///  exitAt / exitKind: where and how the run left its normal path
  ///  recovered:         a recovery action ran
  ///  crashAfterDecision: the process crashed after the COMPLETED decision
  var ghost: { exitAt: str, exitKind: str, recovered: bool, crashAfterDecision: bool }

  // ---------------------------------------------------------------------------
  // Locations
  // ---------------------------------------------------------------------------

  /// All in runtime/_session_engine.py unless stated otherwise.
  /// run.admitted  `SessionEngine.run`, after the `create` lifecycle command
  ///               committed RUNNING with an active profile: `on_admitted`,
  ///               interaction activation and task registration run before the
  ///               setup `try`.
  /// run.setup     `SessionEngine.run`'s setup `try`, which starts with
  ///               `execution_presence.ensure`.
  /// run.started   `SessionEngine.run`'s INTERACTION_STARTED yield.
  /// res.admitted  `SessionEngine.resume`, after `apply_invocation_lifecycle_command`
  ///               admitted the resume and before its setup `try`.
  /// res.setup     `SessionEngine.resume`'s setup `try`, which yields the
  ///               resumed interaction's events.
  /// res.promoted  `SessionEngine.resume` yields the recovered MODEL_COMPLETED in
  ///               its outer forwarding `try`, before the inner stream starts.
  /// body          `SessionEngine._run_session_with_deadline`'s protected `try`;
  ///               it calls `execution_presence.ensure` for resumed runs.
  /// decided       `_complete_session_if_no_queued_messages` committed the
  ///               COMPLETED interaction and session; next comes the
  ///               INTERACTION_COMPLETED yield.
  /// completing    `SessionOutcomePublication.complete_session`
  ///               (runtime/_session_outcome_publication.py), which completes
  ///               the task.
  /// fence         the trailing fence release
  ///               (`EnvironmentLifecycle._release_quiescent_invocation_fence` ->
  ///               `_settle_quiescent_invocation_fence`,
  ///               runtime/_environment_lifecycle.py).
  pure val LOCATIONS = Set(
    "run.create", "run.admitted", "run.setup", "run.started",
    "res.admit", "res.admitted", "res.setup", "res.promoted",
    "body", "decided", "completing", "fence"
  )

  /// Locations suspended at a yield, where the consumer can close the stream.
  pure val YIELDS = Set("run.started", "res.setup", "res.promoted", "body", "decided")

  /// Locations at an await, where cancellation or a grouped cancellation lands.
  pure val AWAITS = Set(
    "run.admitted", "run.setup", "res.admitted", "res.setup",
    "body", "decided", "completing", "fence"
  )

  /// Locations where ordinary setup or body code can raise.
  pure val RAISES = Set("run.admitted", "run.setup", "res.admitted", "res.setup", "body")

  /// What the hand-written handler at each location does on `main`:
  ///  "abandon": finalize abandoned / interrupted: session INTERRUPTED,
  ///             interaction interrupted, fence released;
  ///  "fail":    session FAILED, interaction failed, task failed, fence released;
  ///  "escape":  nothing durable; the error leaves with the session RUNNING
  ///             or the completion half done.
  pure def currentHandler(loc: str, kind: str): str =
    if (loc == "run.admitted")
      // Outside the setup try: nothing owns the admitted run yet.
      "escape"
    else if (loc == "run.setup")
      // Its except CancelledError, except Exception and except
      // BaseExceptionGroup handlers (the group's delivered CancelledError is
      // found) all settle.
      if (kind == "exc") "fail" else "abandon"
    else if (loc == "run.started")
      // except GeneratorExit: shielded handle_session_interrupted.
      "abandon"
    else if (loc == "res.admitted")
      // The admission call's handlers only deactivate the interaction; the
      // code up to the setup try is unprotected.
      "escape"
    else if (loc == "res.setup")
      // except (CancelledError, GeneratorExit) settles; except Exception
      // fails; a BaseExceptionGroup matches neither and reaches except
      // BaseException, which only tries the fence release, and the release is
      // refused without settlement.
      if (kind == "exc") "fail" else if (kind == "group") "escape" else "abandon"
    else if (loc == "res.promoted")
      // The outer except GeneratorExit closes the unstarted inner generator,
      // whose body and handlers never ran; its finally only deactivates.
      "escape"
    else if (loc == "body")
      // except CancelledError, GeneratorExit, Exception and BaseExceptionGroup
      // (binding_finalize_explicit_cancellation) all settle.
      if (kind == "exc") "fail" else "abandon"
    else
      // decided, completing, fence: the interaction is settled, so the
      // `_latest_session_invocation_interaction_is_settled` guards in the
      // body's CancelledError and GeneratorExit handlers re-raise;
      // complete_session and the trailing release never run.
      "escape"

  pure val ACTIVE = LOCATIONS.union(Set("settling"))

  // ---------------------------------------------------------------------------
  // Derived predicates
  // ---------------------------------------------------------------------------

  /// The run is registered in this process (`SessionControl.has_active_tasks`,
  /// checked by `IncompleteSessionRecovery.recover_incomplete_session_scoped`)
  /// and its presence is live.
  val runActive = proc.alive and ACTIVE.contains(proc.pc)

  /// The worker still holds the task lease (heartbeat running).
  val workerLive = proc.alive and proc.pc != "returned" and proc.pc != "dead"

  val isDesign = cfg.variant != "current"

  /// Inactivity sweep: `SessionAbandonment.recover_abandoned`
  /// (runtime/_session_abandonment.py) runs `recover_incomplete_sessions` over
  /// RUNNING and INTERRUPTING; `recover_incomplete_session_scoped`
  /// (runtime/_incomplete_session_recovery.py) skips only a live execution
  /// owner.
  val sweepEnabled = cfg.sweep and db.status == "RUNNING" and not(runActive)

  /// The worker's expired-handoff scanner
  /// (`_recover_expired_interrupted_task_handoffs`, tasks/worker.py): for a
  /// RUNNING session it runs recover_incomplete_session, then hands the task
  /// off only when the session is INTERRUPTED; COMPLETED and FAILED sessions
  /// are skipped.
  val attachedEnabled =
    cfg.hasTask and db.task == "running" and not(workerLive)
      and Set("RUNNING", "INTERRUPTED").contains(db.status)

  /// A claim whose session was never created is unattached: the worker's
  /// `reclaim_expired` returns it to PENDING (`InMemoryTaskStore.reclaim_expired`,
  /// tasks/memory.py).
  val reclaimEnabled = cfg.hasTask and db.task == "running" and not(workerLive) and db.status == "NONE"

  /// design_scan only: recovery also visits COMPLETED sessions with an unsettled
  /// decision (task still running) or an unreleased fence.
  val scanEnabled =
    cfg.variant == "design_scan" and not(runActive) and db.status == "COMPLETED"
      and (db.fenceOpen or (cfg.hasTask and db.task == "running"))

  /// Nothing runs the session and no recovery action has anything left to do.
  val quiescent =
    Set("returned", "dead").contains(proc.pc)
      and not(sweepEnabled) and not(attachedEnabled) and not(scanEnabled) and not(reclaimEnabled)

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  pure def released(d: Db): Db =
    { ...d, fenceOpen: false }

  /// Finalize abandoned / handle interrupted: settles the started interaction as
  /// interrupted and releases the fence.
  pure def abandoned(d: Db): Db =
    released({ ...d, status: "INTERRUPTED", inter: if (d.inter == "started") "interrupted" else d.inter })

  pure def failed(d: Db): Db =
    released({ ...d, status: "FAILED", inter: "failed", task: if (d.task == "running") "failed" else d.task })

  // ---------------------------------------------------------------------------
  // Actions
  // ---------------------------------------------------------------------------

  action init = {
    nondet v = VARIANTS.oneOf()
    nondet entry = Set("run", "resume").oneOf()
    nondet hasTask = Set(true, false).oneOf()
    nondet sweep = Set(true, false).oneOf()
    nondet promoted = Set(true, false).oneOf()
    all {
      cfg' = {
        variant: v,
        hasTask: hasTask,
        // The design reconciles orphaned sessions itself.
        sweep: if (v == "current") sweep else true,
        promoted: entry == "resume" and promoted,
      },
      db' = {
        status: if (entry == "run") "NONE" else "INTERRUPTED",
        fenceOpen: false,
        inter: if (entry == "run") "none" else "interrupted",
        task: if (hasTask) "running" else "none",
        decision: "",
      },
      proc' = {
        alive: true,
        pc: if (entry == "run") "run.create" else "res.admit",
        groupExit: false,
        failExit: false,
      },
      ghost' = { exitAt: "", exitKind: "", recovered: false, crashAfterDecision: false },
    }
  }

  /// Admission: run's `create` lifecycle command or resume's
  /// AdmitInvocationCommand. One write: RUNNING, run epoch + 1, active
  /// profile installed, interaction started.
  action admit = all {
    proc.alive,
    proc.pc == "run.create" or proc.pc == "res.admit",
    db' = { ...db, status: "RUNNING", fenceOpen: true, inter: "started" },
    proc' = { ...proc, pc: if (proc.pc == "run.create") "run.admitted" else "res.admitted" },
    cfg' = cfg, ghost' = ghost,
  }

  /// Normal progress between locations that writes nothing the model tracks
  /// (claiming the execution owner, `execution_presence.ensure`, happens here:
  /// in `SessionEngine.run`'s setup, and in `_run_session_with_deadline` for
  /// resume).
  action advance = all {
    proc.alive,
    Set("run.admitted", "run.setup", "run.started", "res.admitted", "res.setup", "res.promoted").contains(proc.pc),
    val target =
      if (proc.pc == "run.admitted") "run.setup"
      else if (proc.pc == "run.setup") "run.started"
      else if (proc.pc == "run.started") "body"
      else if (proc.pc == "res.admitted") "res.setup"
      else if (proc.pc == "res.setup") (if (cfg.promoted) "res.promoted" else "body")
      else "body"
    all {
      proc' = { ...proc, pc: target },
      db' = db,
      cfg' = cfg, ghost' = ghost,
    }
  }

  /// The body commits the COMPLETED interaction transition; with no workspace
  /// finalization pending this also sets the session COMPLETED
  /// (`_complete_session_if_no_queued_messages`).
  action decide = all {
    proc.alive, proc.pc == "body",
    // The design installs the COMPLETED decision in the same transaction.
    db' = { ...db, status: "COMPLETED", inter: "completed",
            decision: if (isDesign) "completed" else db.decision },
    proc' = { ...proc, pc: "decided" },
    cfg' = cfg, ghost' = ghost,
  }

  /// The consumer takes INTERACTION_COMPLETED; complete_session starts.
  action consumeCompleted = all {
    proc.alive, proc.pc == "decided",
    proc' = { ...proc, pc: "completing" },
    db' = db, cfg' = cfg, ghost' = ghost,
  }

  /// complete_session completes the attached task
  /// (`SessionOutcomePublication.complete_session`) and publishes
  /// SESSION_COMPLETED.
  action completeTask = all {
    proc.alive, proc.pc == "completing",
    db' = { ...db, task: if (db.task == "running") "completed" else db.task },
    proc' = { ...proc, pc: "fence" },
    cfg' = cfg, ghost' = ghost,
  }

  /// Trailing fence release (`_release_quiescent_invocation_fence`); the run
  /// returns normally.
  action releaseFence = all {
    proc.alive, proc.pc == "fence",
    db' = released(db),
    proc' = { ...proc, pc: "exited" },
    cfg' = cfg, ghost' = ghost,
  }

  /// An abnormal exit of the run at its current location.
  action exit(kind: str): bool = all {
    proc.alive,
    if (kind == "close") YIELDS.contains(proc.pc)
    else if (kind == "exc") RAISES.contains(proc.pc)
    else AWAITS.contains(proc.pc),
    ghost' = { ...ghost, exitAt: proc.pc, exitKind: kind },
    cfg' = cfg,
    if (isDesign)
      // RunSettlement.stream / admitted: one handler owns every exit and
      // calls settle(ledger, classify_run_exit(error)).
      all {
        proc' = { ...proc, pc: "settling", groupExit: kind == "group", failExit: kind == "exc" },
        db' = db,
      }
    else
      val h = currentHandler(proc.pc, kind)
      all {
        proc' = { ...proc, pc: "exited", groupExit: kind == "group", failExit: kind == "exc" },
        db' = if (h == "abandon") abandoned(db) else if (h == "fail") failed(db) else db,
      }
  }

  /// The task outcome a decision implies.
  pure def taskFor(decision: str): str =
    if (decision == "completed") "completed" else if (decision == "interrupted") "handoff" else "failed"

  /// The session status a decision implies.
  pure def statusFor(decision: str): str =
    if (decision == "completed") "COMPLETED" else if (decision == "interrupted") "INTERRUPTED" else "FAILED"

  /// All remaining settle() steps at once, as recovery runs them: recovery is
  /// itself idempotent and rerun after a crash, so its own steps need no
  /// interleaving here.
  pure def settleAll(d: Db): Db =
    val decision = if (d.decision != "") d.decision else "interrupted"
    released({ ...d,
      decision: decision,
      task: if (d.task == "running") taskFor(decision) else d.task,
      status: statusFor(decision),
      inter: if (d.inter == "started") decision else d.inter,
    })

  /// design: settle(ledger, exit), one durable step at a time in the fixed
  /// order decide -> task -> terminal status -> fence. Each step is idempotent
  /// and a crash can fall between any two. The FAILED and INTERRUPTED decision
  /// is installed before its side effects with the session still RUNNING
  /// (`claim_invocation_terminal_decision`,
  /// runtime/_session_outcome_publication.py;
  /// `ensure_interruption_terminal_decision`, runtime/_session_finalization.py).
  action settleStep = all {
    proc.alive, proc.pc == "settling",
    cfg' = cfg, ghost' = ghost,
    if (db.decision == "")
      all {
        db' = { ...db, decision: if (proc.failExit) "failed" else "interrupted" },
        proc' = proc,
      }
    else if (cfg.hasTask and db.task == "running")
      all { db' = { ...db, task: taskFor(db.decision) }, proc' = proc }
    else if (db.status == "RUNNING")
      all {
        db' = { ...db, status: statusFor(db.decision),
                inter: if (db.inter == "started") db.decision else db.inter },
        proc' = proc,
      }
    else if (db.fenceOpen)
      all { db' = released(db), proc' = proc }
    else
      all { db' = db, proc' = { ...proc, pc: "returned" } },
  }

  /// The worker's disposition after the handler returns (current code).
  ///  - A grouped cancellation is re-raised unchanged for an ordinary task
  ///    (`_await_task_handler`'s except BaseExceptionGroup, tasks/worker.py)
  ///    and `_handle_with_heartbeat_owned` catches only Exception: the task
  ///    stays held by the exited worker.
  ///  - Otherwise an INTERRUPTED session is handed off
  ///    (`_handoff_interrupted_session`) and anything else still running is
  ///    failed (`_safe_fail`).
  action workerDispose = all {
    proc.alive, proc.pc == "exited",
    proc' = { ...proc, pc: "returned" },
    db' =
      if (cfg.hasTask and db.task == "running" and not(proc.groupExit))
        { ...db, task: if (db.status == "INTERRUPTED") "handoff" else "failed" }
      else db,
    cfg' = cfg, ghost' = ghost,
  }

  /// The process dies; no finally block or background task runs. The owner
  /// row stays and its lease lapses.
  action crash = all {
    proc.alive, proc.pc != "returned",
    proc' = { ...proc, alive: false, pc: "dead" },
    ghost' = { ...ghost, exitAt: if (ghost.exitAt == "") proc.pc else ghost.exitAt,
               exitKind: if (ghost.exitKind == "") "crash" else ghost.exitKind,
               crashAfterDecision: db.status == "COMPLETED" },
    db' = db, cfg' = cfg,
  }

  /// recover_abandoned_sessions / recover_incomplete_sessions on a RUNNING
  /// session with no live owner: finalize abandoned. In the design the same
  /// sweep ends in settle(ledger_from_durable(...)), which follows a durable
  /// decision when there is one and settles the task too.
  action recoverSweep = all {
    sweepEnabled,
    db' = if (isDesign) settleAll(db) else abandoned(db),
    ghost' = { ...ghost, recovered: true },
    proc' = proc, cfg' = cfg,
  }

  /// The task worker's expired-handoff scanner
  /// (`_recover_expired_interrupted_task_handoffs`, tasks/worker.py). It runs
  /// `recover_incomplete_session` on a RUNNING session, which in the design
  /// ends in the same decision-aware settle as the sweep: a recorded FAILED
  /// decision fails the task instead of handing it off.
  action recoverAttached = all {
    attachedEnabled,
    val d = if (db.status == "RUNNING") abandoned(db) else db
    db' = if (isDesign) settleAll(db) else { ...d, task: "handoff" },
    ghost' = { ...ghost, recovered: true },
    proc' = proc, cfg' = cfg,
  }

  /// design_scan: recovery finds the unsettled COMPLETED decision and runs
  /// settle's remaining steps (complete the task, then release the fence).
  action recoverDecision = all {
    scanEnabled,
    db' = settleAll(db),
    ghost' = { ...ghost, recovered: true },
    proc' = proc, cfg' = cfg,
  }

  action reclaimUnattached = all {
    reclaimEnabled,
    db' = { ...db, task: "pending" },
    ghost' = { ...ghost, recovered: true },
    proc' = proc, cfg' = cfg,
  }

  action step = any {
    admit, advance, decide, consumeCompleted, completeTask, releaseFence,
    exit("cancel"), exit("group"), exit("close"), exit("exc"),
    settleStep, workerDispose, crash,
    recoverSweep, recoverAttached, recoverDecision, reclaimUnattached,
    // Stutter at the end so terminal states are not deadlocks.
    all { quiescent, cfg' = cfg, db' = db, proc' = proc, ghost' = ghost },
  }

  // ---------------------------------------------------------------------------
  // Properties
  // ---------------------------------------------------------------------------

  /// No session is RUNNING once nothing runs it and recovery has finished.
  val noOrphanedRunning = quiescent implies db.status != "RUNNING"

  /// The run's own exit settles the session, before any recovery.
  val runExitSettles = (proc.pc == "returned" and not(ghost.recovered)) implies db.status != "RUNNING"

  /// Every started interaction settles.
  val interactionSettles = quiescent implies db.inter != "started"

  /// The task completes exactly when its session completes, and no task stays
  /// running once recovery has finished.
  val taskMatchesSession =
    (quiescent and cfg.hasTask) implies
      (db.task != "running" and ((db.task == "completed") == (db.status == "COMPLETED")))

  /// A worker that has returned holds no task.
  val workerReleasesTask =
    (proc.pc == "returned" and not(ghost.recovered) and cfg.hasTask) implies db.task != "running"

  /// No completed session keeps its run fence (it would refuse every later run
  /// with SessionRunFenced). An INTERRUPTED or FAILED session with an
  /// unreleased fence is not checked: its terminal evidence lets the next
  /// admission or incomplete-session recovery release it
  /// (`EnvironmentLifecycle.release_settled_invocation` and the recovery
  /// paths), and this model does not include those paths.
  val noPermanentFence = quiescent implies not(db.fenceOpen and db.status == "COMPLETED")

  val settled = and {
    noOrphanedRunning, runExitSettles, interactionSettles,
    taskMatchesSession, workerReleasesTask, noPermanentFence,
  }

  // --- Expected to hold -------------------------------------------------------

  /// The single settlement path plus the decision scan satisfies every property.
  val designScanSettles = cfg.variant == "design_scan" implies settled

  /// Without the scan, the design still satisfies every property except after a
  /// crash between the COMPLETED decision and the end of settle().
  val designSettlesWithoutCrashAfterDecision =
    (cfg.variant == "design" and not(ghost.crashAfterDecision)) implies settled

  // --- Known counterexamples on main (expected to be VIOLATED) ----------------
  // Each is the negation of the bad state, scoped to one exit, so the checker
  // reports exactly that trace. They describe the code, not test it: when a
  // fix lands, update `currentHandler` (or the action it changes) to match;
  // the check then reports "no violation", and check.mjs fails until its
  // expectation is updated.

  val current = cfg.variant == "current"

  /// `current` violates each settlement property (expected to be VIOLATED).
  val currentNoOrphanedRunning = current implies noOrphanedRunning
  val currentRunExitSettles = current implies runExitSettles
  val currentInteractionSettles = current implies interactionSettles
  val currentTaskMatchesSession = current implies taskMatchesSession
  val currentWorkerReleasesTask = current implies workerReleasesTask
  val currentNoPermanentFence = current implies noPermanentFence

  /// Grouped cancellation during resume setup leaves the session RUNNING.
  val cexResumeSetupGroupCancel = not(current and ghost.exitAt == "res.setup" and ghost.exitKind == "group"
    and proc.pc == "returned" and not(ghost.recovered) and db.status == "RUNNING")

  /// Cancellation as resume admission
  /// commits leaves the session RUNNING.
  val cexResumeAdmissionCancel = not(current and ghost.exitAt == "res.admitted" and ghost.exitKind == "cancel"
    and proc.pc == "returned" and not(ghost.recovered) and db.status == "RUNNING")

  /// Cancellation right after `create`
  /// commits leaves the session RUNNING with no owner.
  val cexRunEntranceCreate = not(current and ghost.exitAt == "run.admitted" and ghost.exitKind == "cancel"
    and proc.pc == "returned" and not(ghost.recovered) and db.status == "RUNNING")

  /// Closing at the promoted MODEL_COMPLETED leaves the resume RUNNING.
  val cexResumePromotedClose = not(current and ghost.exitAt == "res.promoted" and ghost.exitKind == "close"
    and proc.pc == "returned" and not(ghost.recovered) and db.status == "RUNNING")

  /// Closing at INTERACTION_COMPLETED skips complete_session; the task of
  /// the completed session is failed by the worker instead of completed.
  val cexCompletedInteractionClose = not(current and ghost.exitAt == "decided" and ghost.exitKind == "close"
    and quiescent and cfg.hasTask and db.status == "COMPLETED" and db.task != "completed")

  /// Grouped cancellation between the decision and complete_task leaves the
  /// task running under the completed session forever.
  val cexTaskCompletionGroup = not(current and ghost.exitKind == "group"
    and Set("decided", "completing").contains(ghost.exitAt)
    and quiescent and cfg.hasTask and db.status == "COMPLETED" and db.task == "running")

  /// Grouped cancellation leaves the task held by the exited worker.
  val cexGroupCancelHoldsTask = not(current and ghost.exitKind == "group" and ghost.exitAt == "body"
    and proc.pc == "returned" and not(ghost.recovered) and db.task == "running")

  /// A crash between the decision and
  /// complete_task leaves the task running under the completed session.
  val cexTaskCrashTerminalize = not(current and ghost.exitKind == "crash"
    and Set("decided", "completing").contains(ghost.exitAt)
    and quiescent and db.status == "COMPLETED" and db.task == "running")

  /// Cancellation at the trailing
  /// release leaves the completed session fenced forever.
  val cexTrailingFenceCancel = not(current and ghost.exitKind == "cancel" and ghost.exitAt == "fence"
    and quiescent and db.status == "COMPLETED" and db.fenceOpen)

  /// A crash after the decision leaves the
  /// completed session fenced forever; the sweep skips COMPLETED sessions.
  val cexTrailingFenceCrash = not(current and ghost.exitKind == "crash" and ghost.exitAt == "fence"
    and quiescent and db.status == "COMPLETED" and db.fenceOpen)

  /// A crash leaves a session without an attached task RUNNING forever
  /// when nothing schedules the abandoned-session sweep.
  val cexOrphanedRunningNoSweep = not(current and ghost.exitKind == "crash" and not(cfg.sweep) and not(cfg.hasTask)
    and quiescent and db.status == "RUNNING")

  /// The same with a task (a model prediction, not reproduced yet): an escape leaves the session RUNNING,
  /// the worker's `_safe_fail` fails the task because the session is not
  /// INTERRUPTED, and the expired-handoff scanner then skips the session
  /// because its task is terminal. Only the opt-in sweep would recover it.
  val cexOrphanedRunningTaskFailed = not(current and cfg.hasTask and not(cfg.sweep)
    and quiescent and db.status == "RUNNING" and db.task == "failed")

  /// The design without the decision scan: a crash after the COMPLETED decision
  /// is never finished, because the sweep visits only RUNNING sessions.
  val cexDesignNeedsScan = not(cfg.variant == "design" and ghost.crashAfterDecision
    and quiescent and db.status == "COMPLETED" and (db.fenceOpen or db.task == "running"))

  // --- Reachability (expected to be VIOLATED; guards against a vacuous model) --

  /// The happy path completes the session and its task and releases the fence.
  val reachCompletes = not(proc.pc == "returned" and db.status == "COMPLETED" and not(db.fenceOpen)
    and cfg.hasTask and db.task == "completed" and not(ghost.recovered))

  /// Recovery finishes a run that crashed after its COMPLETED decision.
  val reachRecoveredDecision = not(ghost.crashAfterDecision and ghost.recovered
    and db.status == "COMPLETED" and not(db.fenceOpen) and cfg.hasTask and db.task == "completed")
}

/// One instance per variant; run every check with `--main=<instance>`.
module session_settlement_current {
  import session_settlement(VARIANTS = Set("current")).*
}

module session_settlement_design {
  import session_settlement(VARIANTS = Set("design")).*
}

module session_settlement_design_scan {
  import session_settlement(VARIANTS = Set("design_scan")).*
}
