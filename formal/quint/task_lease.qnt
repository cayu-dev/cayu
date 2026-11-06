// -*- mode: Bluespec; -*-
/**
 * Worker task lease, heartbeat, expiry, reclamation and handoff.
 *
 * One task and two worker incarnations, 1 and 2. An incarnation is one claim's
 * callback with its own heartbeat loop; incarnation 2 is the one that takes the
 * task over after incarnation 1's lease lapses. Both may use the same stable
 * worker name (a restarted or replicated worker with a fixed `worker_id`).
 *
 * Variants:
 *  - "token": the code on `main`. A claim is identified by (worker_id,
 *    lease_expires_at, interrupted_handoff_id); the exact expiry timestamp is
 *    the lease token, and every heartbeat mints a new one (`heartbeat` in
 *    `InMemoryTaskStore`, tasks/memory.py, and `SQLiteTaskStore`,
 *    storage/tasks_sqlite.py). Every claimed mutation checks worker_id plus the
 *    exact token (`_ensure_exact_owned_active_task_lease`, tasks/_lifecycle.py).
 *  - "generation": the proposed generation fence. A claim generation that
 *    claim, reclaim, handoff and release increment and heartbeats keep; every
 *    mutation checks `generation == expected AND lease_expires_at > now`, with
 *    the generation fixed for the callback's lifetime.
 *  - "generation_hb_last": "generation" plus one ordering rule: the heartbeat
 *    stops only after the disposition has committed.
 *
 * Citations name functions under `src/cayu/`.
 *
 * Time is abstract: a lease lapses only while no heartbeat loop of its holder
 * runs (a stalled or stopped loop, or a crashed process). That is the
 * assumption under which a heartbeat loop is meant to keep a lease alive.
 */
module task_lease {
  /// The fencing variants a check explores; the instances at the end of this
  /// file fix one each, which keeps each run small.
  const FENCINGS: Set[str]

  // ---------------------------------------------------------------------------
  // State
  // ---------------------------------------------------------------------------

  /// Configuration, fixed in `init`.
  ///  fencing:     "token" | "generation" | "generation_hb_last"
  ///  sameName:    both incarnations use the same worker_id
  ///  attached:    the task is attached to a session (handoff and the expired-
  ///               handoff scanner apply; reclaim_expired does not)
  ///  retrySeries: the task has a retry policy (heartbeats keep running during
  ///               settlement, `_await_task_handler`; settlement retries
  ///               against the latest acknowledged token,
  ///               `_settle_task_retry_request_with_lease_authority`)
  var cfg: { fencing: str, sameName: bool, attached: bool, retrySeries: bool }

  /// The task row.
  ///  status:  "PENDING" | "CLAIMED" | "HANDOFF" | "COMPLETED" | "FAILED"
  ///           (CLAIMED stands for CLAIMED and RUNNING; HANDOFF is an attached
  ///           task released for an interrupted-session continuation)
  ///  owner:   worker_id of the claim, "" when unclaimed
  ///  tok:     the lease token (exact lease_expires_at), 0 when unclaimed
  ///  live:    the lease has not expired
  ///  gen:     claim generation (generation variants only)
  ///  nextTok: the next token value the clock will mint
  var task: { status: str, owner: str, tok: int, live: bool, gen: int, nextTok: int }

  /// One record per incarnation.
  ///  pc:       "idle" | "run" | "settle" | "settleSent" | "settleRejected" |
  ///            "hoRead" | "hoCommit" | "finish" | "done" | "crashed"
  ///  outcome:  what the handler returned: "" | "complete" | "fail" | "handoff"
  ///  snapTok:  the token this incarnation believes it holds (lease authority)
  ///  ackTok:   a committed renewal whose acknowledgement has not been applied
  ///  reqTok:   the token in the in-flight settlement request
  ///  readTok:  the token the handoff preflight read from the row
  ///  snapGen:  the generation this incarnation claimed (generation variants)
  ///  hb:       its heartbeat loop runs
  ///  tries:    settlement attempts (at most 3,
  ///            `_settle_task_retry_request_with_lease_authority`)
  ///  recorded: its outcome committed
  ///  stalled:  its heartbeat loop stalled while the callback ran (a genuine
  ///            loss of authority, not a bug)
  ///  expiredRefusal: the store refused one of its dispositions because the
  ///            lease had expired
  var inc: int -> {
    pc: str, outcome: str, snapTok: int, ackTok: int, reqTok: int, readTok: int,
    snapGen: int, hb: bool, tries: int, recorded: bool, stalled: bool, expiredRefusal: bool
  }

  /// History.
  ///  holder:     the incarnation whose claim is current, 0 when unclaimed
  ///  staleWrite: a mutation committed by an incarnation that is not the holder
  ///  falseLoss:  the store refused the holder while its lease was live
  var ghost: { holder: int, staleWrite: bool, falseLoss: bool }

  pure val INCS = Set(1, 2)
  pure val MAX_TOK = 7
  pure val MAX_TRIES = 3

  pure def name(c: int, sameName: bool): str =
    if (sameName) "w" else if (c == 1) "w1" else "w2"

  val usesGen = cfg.fencing != "token"
  val hbLast = cfg.fencing == "generation_hb_last"

  /// The store's authority check for a claimed mutation by incarnation `c`
  /// presenting token `t`.
  ///  token:      worker_id matches, the lease is live and the token is exact
  ///              (`_ensure_exact_owned_active_task_lease`; the same check in
  ///              `InMemoryTaskStore.heartbeat`)
  ///  generation: the generation the callback claimed is still current and the
  ///              lease is live (the proposed generation fence)
  def authorized(c: int, t: int): bool =
    if (usesGen) task.live and task.gen == inc.get(c).snapGen and task.owner != ""
    else task.owner == name(c, cfg.sameName) and task.live and task.tok == t

  val active = Set("run", "settle", "settleSent", "settleRejected", "hoRead", "hoCommit", "finish")

  /// The holder's heartbeat keeps the lease alive.
  val holderHeartbeating = ghost.holder != 0 and inc.get(ghost.holder).hb

  /// The holder after incarnation `c` releases or terminalizes the claim. A
  /// stale write by someone else does not end the holder's claim in the
  /// history: its callback still runs and still believes it owns the task.
  def holderAfterRelease(c: int): int = if (ghost.holder == c) 0 else ghost.holder

  // ---------------------------------------------------------------------------
  // Actions
  // ---------------------------------------------------------------------------

  pure val IDLE = {
    pc: "idle", outcome: "", snapTok: 0, ackTok: 0, reqTok: 0, readTok: 0,
    snapGen: 0, hb: false, tries: 0, recorded: false, stalled: false,
    expiredRefusal: false
  }

  action init = {
    nondet fencing = FENCINGS.oneOf()
    nondet sameName = Set(true, false).oneOf()
    nondet attached = Set(true, false).oneOf()
    nondet retrySeries = Set(true, false).oneOf()
    all {
      cfg' = { fencing: fencing, sameName: sameName, attached: attached, retrySeries: retrySeries },
      task' = { status: "PENDING", owner: "", tok: 0, live: false, gen: 0, nextTok: 1 },
      inc' = INCS.mapBy(_ => IDLE),
      ghost' = { holder: 0, staleWrite: false, falseLoss: false },
    }
  }

  /// `claim_task` (`InMemoryTaskStore`, tasks/memory.py) for PENDING, or
  /// claim_interrupted_task_continuation for HANDOFF. Generation + 1.
  /// Incarnation 2 only takes over after incarnation 1 has claimed.
  action claim(c: int): bool = all {
    inc.get(c).pc == "idle",
    c == 1 or inc.get(1).pc != "idle",
    task.status == "PENDING" or task.status == "HANDOFF",
    task.nextTok <= MAX_TOK,
    task' = { ...task, status: "CLAIMED", owner: name(c, cfg.sameName), tok: task.nextTok,
              live: true, gen: task.gen + 1, nextTok: task.nextTok + 1 },
    inc' = inc.set(c, { ...IDLE, pc: "run", snapTok: task.nextTok, snapGen: task.gen + 1, hb: true }),
    ghost' = { ...ghost, holder: c },
    cfg' = cfg,
  }

  /// A heartbeat reaches the store with the incarnation's current token
  /// (`_heartbeat_until`, tasks/worker.py). Token: a new expiry is minted. Generation:
  /// the expiry moves, the generation stays. A refused heartbeat ends the loop
  /// with lease-authority loss.
  action heartbeat(c: int): bool = all {
    val i = inc.get(c)
    all {
      i.hb, active.contains(i.pc), i.ackTok == 0, task.nextTok <= MAX_TOK,
      if (authorized(c, i.snapTok))
        all {
          task' = { ...task, tok: task.nextTok, nextTok: task.nextTok + 1 },
          inc' = inc.set(c, { ...i, ackTok: task.nextTok }),
          ghost' = { ...ghost, staleWrite: ghost.staleWrite or ghost.holder != c },
        }
      else
        all {
          task' = task,
          inc' = inc.set(c, { ...i, hb: false }),
          ghost' = { ...ghost, falseLoss: ghost.falseLoss or (ghost.holder == c and task.live) },
        },
      cfg' = cfg,
    }
  }

  /// The heartbeat's acknowledgement updates the lease authority, under its
  /// lock (`_heartbeat_until`). It can arrive late: any write sent in
  /// between carries the previous token.
  action heartbeatAck(c: int): bool = all {
    val i = inc.get(c)
    all {
      i.ackTok != 0,
      inc' = inc.set(c, { ...i, snapTok: i.ackTok, ackTok: 0 }),
      task' = task, ghost' = ghost, cfg' = cfg,
    }
  }

  /// The heartbeat loop stalls while the callback runs (a long pause or a
  /// partition): authority may be lost for real.
  action stall(c: int): bool = all {
    val i = inc.get(c)
    all {
      i.hb, i.pc == "run",
      inc' = inc.set(c, { ...i, hb: false, stalled: true }),
      task' = task, ghost' = ghost, cfg' = cfg,
    }
  }

  /// The lease lapses once nothing renews it.
  action expire = all {
    task.live, task.status == "CLAIMED", not(holderHeartbeating),
    task' = { ...task, live: false },
    inc' = inc, ghost' = ghost, cfg' = cfg,
  }

  /// Unattached: reclaim_expired returns the claim to PENDING
  /// (`InMemoryTaskStore.reclaim_expired`, called from `run_task_worker`).
  /// Attached: the expired-handoff scanner releases it for a continuation
  /// (`_recover_expired_interrupted_task_handoffs`, tasks/worker.py).
  /// Either way the generation moves on.
  action reclaim = all {
    task.status == "CLAIMED", not(task.live),
    task' = { ...task, status: if (cfg.attached) "HANDOFF" else "PENDING", owner: "", tok: 0,
              gen: task.gen + 1 },
    ghost' = { ...ghost, holder: 0 },
    inc' = inc, cfg' = cfg,
  }

  /// The handler returns. Ordinary tasks stop the heartbeat before the
  /// disposition (`stop_heartbeat.set()` in `_await_task_handler`,
  /// tasks/worker.py); retry-series tasks keep it; the hb_last variant keeps
  /// it until the disposition commits.
  action handlerReturns(c: int): bool = all {
    val i = inc.get(c)
    all {
      i.pc == "run",
      nondet outcome = (if (cfg.attached) Set("complete", "handoff") else Set("complete", "fail")).oneOf()
      inc' = inc.set(c, { ...i,
        outcome: outcome,
        pc: if (outcome == "handoff") "hoRead" else "settle",
        hb: i.hb and (cfg.retrySeries or hbLast),
      }),
      task' = task, ghost' = ghost, cfg' = cfg,
    }
  }

  /// Once the heartbeat loop has been stopped, the disposition starts only after
  /// the loop has ended: `_await_task_handler` sets `stop_heartbeat` and awaits
  /// the heartbeat task, which applies any pending acknowledgement, and the
  /// ordinary disposition runs under `lease_authority.lock`. With the loop still
  /// running (retry-series tasks, or hb_last) a renewal can be in flight.
  def heartbeatJoined(c: int): bool = inc.get(c).hb or inc.get(c).ackTok == 0

  /// A settlement attempt snapshots the current lease authority and goes to
  /// the store; for retry-series tasks outside the renewal lock
  /// (`_settle_retry_attempt_report`,
  /// `_settle_task_retry_request_with_lease_authority`).
  action settleSend(c: int): bool = all {
    val i = inc.get(c)
    all {
      i.pc == "settle", heartbeatJoined(c),
      inc' = inc.set(c, { ...i, pc: "settleSent", reqTok: i.snapTok, tries: i.tries + 1 }),
      task' = task, ghost' = ghost, cfg' = cfg,
    }
  }

  /// The store applies or refuses the terminalization.
  action settleCommit(c: int): bool = all {
    val i = inc.get(c)
    all {
      i.pc == "settleSent",
      if (authorized(c, i.reqTok) and (task.status == "CLAIMED"))
        all {
          task' = { ...task, status: if (i.outcome == "complete") "COMPLETED" else "FAILED",
                    owner: "", tok: 0 },
          inc' = inc.set(c, { ...i, pc: "finish", recorded: true }),
          ghost' = { ...ghost, holder: holderAfterRelease(c), staleWrite: ghost.staleWrite or ghost.holder != c },
        }
      else
        all {
          task' = task,
          inc' = inc.set(c, { ...i, pc: "settleRejected",
                              expiredRefusal: i.expiredRefusal or not(task.live) }),
          ghost' = { ...ghost, falseLoss: ghost.falseLoss or (ghost.holder == c and task.live) },
        },
      cfg' = cfg,
    }
  }

  /// After TaskClaimLost: wait for a pending renewal acknowledgement, then
  /// retry only if the acknowledged token changed
  /// (`_settle_task_retry_request_with_lease_authority`),
  /// at most three times. Ordinary tasks do not retry: `_safe_fail_payload`
  /// suppresses the loss and the worker returns normally.
  action settleAfterRejection(c: int): bool = all {
    val i = inc.get(c)
    all {
      i.pc == "settleRejected", i.ackTok == 0,
      val retry = cfg.retrySeries and i.snapTok != i.reqTok and i.tries < MAX_TRIES
      inc' = inc.set(c, { ...i, pc: if (retry) "settle" else "finish" }),
      task' = task, ghost' = ghost, cfg' = cfg,
    }
  }

  /// `_handoff_interrupted_session` preflight (tasks/worker.py):
  /// reload the task, require the same worker_id and a lease, and build the
  /// request from the row it just read (`interrupted_task_handoff_request`), not from the callback's own
  /// authority. The generation variants keep the callback's generation.
  action handoffRead(c: int): bool = all {
    val i = inc.get(c)
    all {
      i.pc == "hoRead", heartbeatJoined(c),
      val ok = task.status == "CLAIMED" and task.owner == name(c, cfg.sameName) and task.tok != 0
      inc' = inc.set(c, { ...i, pc: if (ok) "hoCommit" else "finish", readTok: task.tok }),
      task' = task, ghost' = ghost, cfg' = cfg,
    }
  }

  /// The handoff releases the claim for a continuation
  /// (`_settle_interrupted_task_handoff_with_retry`).
  action handoffCommit(c: int): bool = all {
    val i = inc.get(c)
    all {
      i.pc == "hoCommit",
      if (authorized(c, i.readTok) and task.status == "CLAIMED")
        all {
          task' = { ...task, status: "HANDOFF", owner: "", tok: 0, gen: task.gen + 1 },
          inc' = inc.set(c, { ...i, pc: "finish", recorded: true }),
          ghost' = { ...ghost, holder: holderAfterRelease(c), staleWrite: ghost.staleWrite or ghost.holder != c },
        }
      else
        all {
          task' = task,
          inc' = inc.set(c, { ...i, pc: "finish",
                              expiredRefusal: i.expiredRefusal or not(task.live) }),
          ghost' = { ...ghost, falseLoss: ghost.falseLoss or (ghost.holder == c and task.live) },
        },
      cfg' = cfg,
    }
  }

  /// The disposition is over: the heartbeat stops and the worker returns.
  action finish(c: int): bool = all {
    val i = inc.get(c)
    all {
      i.pc == "finish", i.ackTok == 0,
      inc' = inc.set(c, { ...i, pc: "done", hb: false }),
      task' = task, ghost' = ghost, cfg' = cfg,
    }
  }

  /// The process of incarnation `c` dies.
  action crash(c: int): bool = all {
    val i = inc.get(c)
    all {
      active.contains(i.pc),
      inc' = inc.set(c, { ...i, pc: "crashed", hb: false, ackTok: 0 }),
      task' = task, ghost' = ghost, cfg' = cfg,
    }
  }

  action step = any {
    nondet c = INCS.oneOf()
    any {
      claim(c), heartbeat(c), heartbeatAck(c), stall(c), handlerReturns(c),
      settleSend(c), settleCommit(c), settleAfterRejection(c),
      handoffRead(c), handoffCommit(c), finish(c), crash(c),
    },
    expire, reclaim,
    all { task' = task, inc' = inc, ghost' = ghost, cfg' = cfg },
  }

  // ---------------------------------------------------------------------------
  // Properties
  // ---------------------------------------------------------------------------

  /// Only the holder of the current claim changes the task.
  val noStaleOwnerWrite = not(ghost.staleWrite)

  /// The store never refuses the holder while its lease is live.
  val noFalseClaimLost = not(ghost.falseLoss)

  /// A callback that kept its heartbeat and was never displaced has its
  /// outcome recorded by the time its worker returns.
  val reportRecorded = INCS.forall(c =>
    val i = inc.get(c)
    (i.pc == "done" and not(i.stalled)) implies i.recorded)

  /// A live callback is never released or terminalized by someone else.
  val liveHolderKeepsClaim = INCS.forall(c =>
    val i = inc.get(c)
    (i.pc == "run" and ghost.holder == c) implies task.status == "CLAIMED")

  val safe = and { noStaleOwnerWrite, noFalseClaimLost, reportRecorded, liveHolderKeepsClaim }

  // --- Expected to hold -------------------------------------------------------

  /// Generation fencing with the heartbeat stopped last satisfies everything.
  val generationHbLastSafe = cfg.fencing == "generation_hb_last" implies safe

  /// Generation fencing alone already removes stale-owner writes and false
  /// claim losses.
  val generationFences = cfg.fencing == "generation" implies
    (noStaleOwnerWrite and noFalseClaimLost and liveHolderKeepsClaim)

  // --- Known counterexamples on main (expected to be VIOLATED) ----------------
  // They describe the code, not test it: when a fix lands, update the
  // "token" variant to match; the check then reports "no violation", and
  // check.mjs fails until its expectation is updated.

  val tok = cfg.fencing == "token"

  /// `token` violates each lease property (expected to be VIOLATED).
  val tokenNoStaleOwnerWrite = tok implies noStaleOwnerWrite
  val tokenNoFalseClaimLost = tok implies noFalseClaimLost
  val tokenReportRecorded = tok implies reportRecorded
  val tokenLiveHolderKeepsClaim = tok implies liveHolderKeepsClaim

  /// A delayed handoff of incarnation 1 releases incarnation 2, which
  /// claimed under the same worker_id and whose callback still runs.
  val cexHandoffReleasesSameNameReplacement = not(tok and cfg.sameName and cfg.attached
    and inc.get(2).pc == "run" and ghost.staleWrite)

  /// A retry-series settlement loses three times to its own
  /// heartbeats; the worker raises TaskClaimLost with the lease still live and
  /// the report unrecorded.
  val cexRetrySettlementLosesToHeartbeats = not(tok and cfg.retrySeries and not(cfg.attached)
    and not(inc.get(1).stalled) and inc.get(1).pc == "done" and not(inc.get(1).recorded)
    and task.live)

  /// The ordinary-task heartbeat stops before the disposition; the lease
  /// lapses and the failure is refused for that expiry, so the worker returns
  /// with the outcome unrecorded.
  val cexHeartbeatStoppedBeforeDisposition = not(tok and not(cfg.retrySeries) and not(cfg.attached)
    and not(inc.get(1).stalled) and inc.get(1).pc == "done" and not(inc.get(1).recorded)
    and inc.get(1).expiredRefusal)

  /// Generation fencing without "heartbeat last" still loses that outcome to
  /// the expiry.
  val cexGenerationNeedsHbLast = not(cfg.fencing == "generation" and not(cfg.retrySeries)
    and not(inc.get(1).stalled) and inc.get(1).pc == "done" and not(inc.get(1).recorded)
    and inc.get(1).expiredRefusal)

  // --- Reachability (expected to be VIOLATED; guards against a vacuous model) --

  /// Incarnation 2 can take over and complete under generation fencing.
  val reachTakeoverCompletes = not(cfg.fencing == "generation_hb_last" and task.status == "COMPLETED"
    and inc.get(2).recorded)

  /// A handoff can commit under generation fencing.
  val reachHandoff = not(cfg.fencing == "generation_hb_last" and inc.get(1).recorded
    and inc.get(1).outcome == "handoff")
}

/// One instance per variant; run every check with `--main=<instance>`.
module task_lease_token {
  import task_lease(FENCINGS = Set("token")).*
}

module task_lease_generation {
  import task_lease(FENCINGS = Set("generation")).*
}

module task_lease_generation_hb_last {
  import task_lease(FENCINGS = Set("generation_hb_last")).*
}
