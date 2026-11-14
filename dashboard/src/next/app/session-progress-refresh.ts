/**
 * What a running session's progress ticks refresh, and how often.
 *
 * A tick says the session has new events. Only the session's own views are refreshed,
 * never global lists such as the Inbox, and only the views the new events can change
 * when the follower reports their types. Ticks are coalesced: what changes the session's
 * state and the view on screen are refetched at most once per
 * `SESSION_REFRESH_INTERVAL_MS`, and secondary progress (totals, related sessions, a
 * transcript behind the Trace) at most once per `SESSION_SECONDARY_REFRESH_INTERVAL_MS`.
 * A terminal drain refreshes everything at once.
 */

import { inboxKeys } from "../../parts/inbox/keys.ts"
import { partKeys } from "../../parts/keys.ts"

export const SESSION_REFRESH_INTERVAL_MS = 2000
export const SESSION_SECONDARY_REFRESH_INTERVAL_MS = 10_000

export const SESSION_VIEWS = [
  "session",
  "overview",
  "state",
  "session-state",
  "session-children",
  "session-parent",
  "session-messages",
  "queued-messages",
  "session-recovery",
  "inbox",
  "trace",
  "trace-events",
  "transcript",
] as const

export type SessionView = (typeof SESSION_VIEWS)[number]

/**
 * What a batch of new events changes. `urgent` views can change what the reader may do
 * (the session's state, a pending approval, queued messages) and refresh at the fast
 * pace. `progress` views refresh at the fast pace while on screen and at the slow pace
 * otherwise (`isSecondaryView`).
 */
export type RefreshPlan = { urgent: Set<SessionView>; progress: Set<SessionView> }

type Change = { urgent?: readonly SessionView[]; progress?: readonly SessionView[] }

const TRACE: readonly SessionView[] = ["trace", "trace-events"]
const LIFECYCLE: Change = {
  urgent: ["session", "overview", "state", "session-state", "session-recovery", "inbox"],
  progress: [...TRACE, "transcript"],
}
const ATTENTION: Change = {
  urgent: ["overview", "session-recovery", "inbox"],
  progress: TRACE,
}

/**
 * What a single event type changes. The families that move a session's state (session,
 * interaction, recovery, workflow, task) and its approvals are named here; every other
 * type (requests, context, budgets, providers, environments...) is progress in the
 * session's trace and totals.
 */
export function changeForEventType(type: string): Change {
  if (type === "model.text.delta" || type === "model.thinking.delta") return { progress: TRACE }
  // model.started follows any tool results the previous round published.
  if (type === "model.completed" || type === "turn.completed" || type === "model.started")
    return { progress: ["overview", "transcript", ...TRACE] }
  if (type.startsWith("model.")) return { progress: ["overview", ...TRACE] }
  if (
    type === "tool.call.approval_requested" ||
    type === "tool.call.approved" ||
    type === "tool.call.approval_denied" ||
    type === "tool.call.approval_expired" ||
    type === "tool.call.blocked" ||
    type === "session.awaiting_user_input"
  )
    return ATTENTION
  if (
    type.startsWith("tool.") ||
    type.startsWith("runner.") ||
    type.startsWith("structured_output.")
  )
    return { progress: ["overview", ...TRACE] }
  if (type === "session.delegated_action.updated")
    return { urgent: ["session-children", ...ATTENTION.urgent!], progress: TRACE }
  if (type.startsWith("session.message."))
    return { urgent: ["session-messages"], progress: ["transcript", ...TRACE] }
  if (type === "session.forked") return { progress: ["session-children", ...TRACE] }
  if (
    type.startsWith("session.") ||
    type.startsWith("interaction.") ||
    type.startsWith("recovery.")
  )
    return LIFECYCLE
  if (type.startsWith("workflow.") || type.startsWith("task."))
    return { progress: ["session-children", "overview", ...TRACE] }
  return { progress: ["overview", ...TRACE] }
}

/**
 * What a tick changes when the follower reports only sequences. Urgently: the overview,
 * which carries the state that says whether the session runs, waits or stopped, and the
 * queued messages while any is still queued (delivery happens mid-run). As progress: the
 * trace, transcript and related sessions, and the session's detail and runtime state.
 * A stop or a wait then drains every view.
 */
export const UNTYPED_CHANGE: Change = {
  urgent: ["overview", "queued-messages"],
  progress: [...TRACE, "transcript", "session-children", "session", "state", "session-state"],
}

/** What a batch of new events changes. A terminal drain changes every view, urgently. */
export function planForProgress(
  eventTypes: readonly string[] | null | undefined,
  terminal = false,
): RefreshPlan {
  const plan: RefreshPlan = { urgent: new Set(), progress: new Set() }
  const changes = terminal
    ? [{ urgent: SESSION_VIEWS }]
    : eventTypes
      ? eventTypes.map(changeForEventType)
      : [UNTYPED_CHANGE]
  for (const change of changes) {
    for (const view of change.urgent ?? []) plan.urgent.add(view)
    for (const view of change.progress ?? []) plan.progress.add(view)
  }
  for (const view of plan.urgent) plan.progress.delete(view)
  return plan
}

/**
 * Whether a progress view is secondary for the reader right now: refreshed at the slow
 * pace. The trace and the transcript are primary only in their own view.
 */
export function isSecondaryView(view: SessionView, conversation: boolean): boolean {
  if (view === "transcript") return !conversation
  if (view === "trace" || view === "trace-events") return conversation
  return true
}

type Timer = ReturnType<typeof setTimeout>

/**
 * Coalesces ticks into at most one flush per interval. The first tick after a quiet
 * interval flushes at once; ticks inside the interval are merged into one trailing
 * flush. `flushNow` (a terminal drain) skips the wait. Flushes never overlap: views
 * queued while one runs wait for it, so a new flush doesn't cancel slow reads.
 */
export class SessionRefreshThrottle {
  readonly #flush: (views: Set<SessionView>) => Promise<void> | void
  readonly #interval: number
  readonly #now: () => number
  #pending = new Set<SessionView>()
  #last = Number.NEGATIVE_INFINITY
  #timer: Timer | null = null
  #inflight: Promise<void> | null = null
  #disposed = false

  constructor(
    flush: (views: Set<SessionView>) => Promise<void> | void,
    { interval = SESSION_REFRESH_INTERVAL_MS, now = () => Date.now() } = {},
  ) {
    this.#flush = flush
    this.#interval = interval
    this.#now = now
  }

  /** Queue views; flushes now or once the interval since the last flush has passed. */
  add(views: Iterable<SessionView>): Promise<void> | void {
    if (this.#disposed) return
    for (const view of views) this.#pending.add(view)
    return this.#schedule()
  }

  #schedule(): Promise<void> | void {
    if (this.#pending.size === 0 || this.#timer || this.#inflight) return
    const wait = this.#last + this.#interval - this.#now()
    if (wait <= 0) return this.#run()
    this.#timer = setTimeout(() => {
      this.#timer = null
      void this.#run()
    }, wait)
  }

  flushNow(views: Iterable<SessionView>): Promise<void> | void {
    if (this.#disposed) return
    for (const view of views) this.#pending.add(view)
    if (this.#timer) clearTimeout(this.#timer)
    this.#timer = null
    if (this.#inflight) return this.#inflight.then(() => this.#run())
    return this.#run()
  }

  dispose(): void {
    this.#disposed = true
    if (this.#timer) clearTimeout(this.#timer)
    this.#timer = null
    this.#pending.clear()
  }

  #run(): Promise<void> | void {
    if (this.#disposed || this.#pending.size === 0 || this.#inflight) return
    const views = this.#pending
    this.#pending = new Set()
    this.#last = this.#now()
    const result = this.#flush(views)
    if (!result || typeof result.then !== "function") return result
    const settled = result.finally(() => {
      this.#inflight = null
      if (!this.#disposed) void this.#schedule()
    })
    this.#inflight = settled.catch(() => undefined)
    return settled
  }
}

type Invalidator = {
  invalidateQueries(filters: {
    queryKey: readonly unknown[]
    predicate?: (query: {
      queryKey: readonly unknown[]
      meta?: Record<string, unknown>
      state?: { data?: unknown }
    }) => boolean
  }): Promise<void>
}

/** Whether a loaded queued-message inspection still holds a message waiting for delivery. */
export function hasQueuedMessage(data: unknown): boolean {
  const pages = (data as { pages?: { records?: { status?: string | null }[] }[] } | undefined)
    ?.pages
  return !!pages?.some((page) => page.records?.some((record) => record.status === "queued"))
}

/**
 * Refetch the given views of one session. The session's Inbox card is refreshed through
 * its session-scoped query; the global Inbox and its counts are left alone. The
 * transcript is refreshed only while the reader follows the latest messages.
 */
export function refreshSessionViews(
  client: Invalidator,
  sessionId: string,
  views: Iterable<SessionView>,
  following: boolean,
): Promise<void> {
  const work: Promise<void>[] = []
  for (const view of views) {
    if (view === "transcript" && !following) continue
    if (view === "inbox")
      // The session's own Inbox card, never the global list or its counts.
      work.push(
        client.invalidateQueries({
          queryKey: [...inboxKeys.all, "list"],
          predicate: (query) =>
            (query.queryKey[3] as { session_id?: string } | undefined)?.session_id === sessionId,
        }),
      )
    else if (view === "queued-messages")
      // The queue panel, only while it shows a message still waiting for delivery.
      work.push(
        client.invalidateQueries({
          queryKey: ["session-messages", sessionId],
          predicate: (query) => hasQueuedMessage(query.state?.data),
        }),
      )
    else if (view === "session-recovery")
      work.push(client.invalidateQueries({ queryKey: partKeys.sessionRecoveries(sessionId) }))
    else if (view === "trace-events")
      work.push(
        client.invalidateQueries({
          queryKey: ["trace-events", sessionId],
          predicate: (query) => query.meta?.sessionProgress !== false,
        }),
      )
    else work.push(client.invalidateQueries({ queryKey: [view, sessionId] }))
  }
  return Promise.all(work).then(() => undefined)
}

/**
 * Two throttles, one per pace: ticks are split into the views on screen (every 2 s at
 * most) and secondary views (every 10 s at most). `drain` flushes both now.
 */
export class SessionTickRefresher {
  readonly #primary: SessionRefreshThrottle
  readonly #secondary: SessionRefreshThrottle

  constructor(
    flush: (views: Set<SessionView>) => Promise<void> | void,
    {
      interval = SESSION_REFRESH_INTERVAL_MS,
      secondaryInterval = SESSION_SECONDARY_REFRESH_INTERVAL_MS,
      now = () => Date.now(),
    } = {},
  ) {
    this.#primary = new SessionRefreshThrottle(flush, { interval, now })
    this.#secondary = new SessionRefreshThrottle(flush, { interval: secondaryInterval, now })
  }

  tick(plan: RefreshPlan, conversation: boolean): Promise<void> {
    const primary: SessionView[] = [...plan.urgent]
    const secondary: SessionView[] = []
    for (const view of plan.progress)
      (isSecondaryView(view, conversation) ? secondary : primary).push(view)
    return Promise.all([this.#primary.add(primary), this.#secondary.add(secondary)]).then(
      () => undefined,
    )
  }

  drain(views: Iterable<SessionView>): Promise<void> {
    const all = [...views]
    return Promise.all([this.#primary.flushNow(all), this.#secondary.flushNow([])]).then(
      () => undefined,
    )
  }

  dispose(): void {
    this.#primary.dispose()
    this.#secondary.dispose()
  }
}
