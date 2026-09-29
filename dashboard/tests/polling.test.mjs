import assert from "node:assert/strict"
import test from "node:test"

import {
  environmentManager,
  QueryClient,
  QueryObserver,
  timeoutManager,
} from "@tanstack/react-query"

import {
  activityPollInterval,
  IDLE_POLL_INTERVAL_MS,
  operationalSnapshotIsActive,
  responseFingerprint,
  sessionStatusIsActive,
  taskStatusIsActive,
  UnchangedResponseCounter,
  waitingPollInterval,
} from "../src/lib/polling.ts"

function snapshot(sessionCounts, taskCounts = null) {
  const zeroSessions = {
    pending: "0",
    running: "0",
    interrupting: "0",
    interrupted: "0",
    completed: "0",
    failed: "0",
  }
  const zeroTasks = {
    pending: "0",
    claimed: "0",
    running: "0",
    paused: "0",
    blocked: "0",
    needs_attention: "0",
    completed: "0",
    failed: "0",
    cancelled: "0",
  }
  return {
    sessions: { counts_by_status: { ...zeroSessions, ...sessionCounts } },
    tasks: taskCounts === null ? null : { counts_by_status: { ...zeroTasks, ...taskCounts } },
  }
}

async function settle() {
  for (let index = 0; index < 5; index += 1) await Promise.resolve()
}

// A manual clock for Query Core's timers. Node 22's mocked setInterval keeps
// re-arming an interval that its own callback clears, which Query Core does
// whenever a poll starts, so the built-in timer mocks fire stale intervals there.
class ManualTimers {
  now = 0
  #nextId = 1
  #timers = new Map()

  setTimeout = (callback, delay) => this.#schedule(callback, delay, false)
  setInterval = (callback, delay) => this.#schedule(callback, delay, true)
  clearTimeout = (id) => this.#timers.delete(id)
  clearInterval = (id) => this.#timers.delete(id)

  #schedule(callback, delay, repeat) {
    const id = this.#nextId
    this.#nextId += 1
    const interval = Math.max(1, delay ?? 0)
    this.#timers.set(id, { callback, due: this.now + interval, interval, repeat })
    return id
  }

  advance(milliseconds) {
    const target = this.now + milliseconds
    for (;;) {
      let nextId = null
      let next = null
      for (const [id, timer] of this.#timers) {
        if (timer.due <= target && (next === null || timer.due < next.due)) {
          nextId = id
          next = timer
        }
      }
      if (next === null) break
      this.now = next.due
      // Re-arm before the callback so a callback that clears its interval stops it.
      if (next.repeat) next.due += next.interval
      else this.#timers.delete(nextId)
      next.callback()
    }
    this.now = target
  }
}

test("overview pages poll quickly only while something is active", () => {
  assert.equal(activityPollInterval(true, 5000), 5000)
  assert.equal(activityPollInterval(false, 5000), IDLE_POLL_INTERVAL_MS)
  assert.ok(IDLE_POLL_INTERVAL_MS >= 30_000)

  for (const status of ["pending", "running", "interrupting"]) {
    assert.equal(sessionStatusIsActive(status), true, status)
  }
  for (const status of ["interrupted", "completed", "failed"]) {
    assert.equal(sessionStatusIsActive(status), false, status)
  }
  for (const status of ["pending", "claimed", "running"]) {
    assert.equal(taskStatusIsActive(status), true, status)
  }
  for (const status of [
    "waiting_dependencies",
    "waiting_group",
    "dependency_skipped",
    "paused",
    "blocked",
    "needs_attention",
    "completed",
    "failed",
    "cancelled",
  ]) {
    assert.equal(taskStatusIsActive(status), false, status)
  }
})

test("an operational snapshot is active while sessions or tasks are in flight", () => {
  assert.equal(operationalSnapshotIsActive(undefined), false)
  assert.equal(operationalSnapshotIsActive(snapshot({ completed: "40", failed: "2" })), false)
  assert.equal(operationalSnapshotIsActive(snapshot({ interrupted: "3" }, {})), false)
  assert.equal(operationalSnapshotIsActive(snapshot({ running: "1" })), true)
  assert.equal(operationalSnapshotIsActive(snapshot({ pending: "1" })), true)
  assert.equal(operationalSnapshotIsActive(snapshot({ interrupting: "1" })), true)
  assert.equal(operationalSnapshotIsActive(snapshot({}, { claimed: "1" })), true)
  assert.equal(operationalSnapshotIsActive(snapshot({}, { blocked: "4", completed: "9" })), false)
})

test("waiting pages back off from five seconds to a one-minute cap", () => {
  assert.deepEqual(
    [0, 1, 2, 3, 4, 100].map(waitingPollInterval),
    [5000, 10_000, 30_000, 60_000, 60_000, 60_000],
  )
  assert.equal(waitingPollInterval(-1), 5000)
})

test("unchanged responses are counted once per response and reset on change", () => {
  const counter = new UnchangedResponseCounter()
  const same = responseFingerprint({ status: "interrupted", updated_at: "t1" })
  const changed = responseFingerprint({ status: "interrupted", updated_at: "t2" })

  assert.equal(counter.observe("q", 0, null), 0)
  assert.equal(counter.observe("q", 1, same), 0)
  assert.equal(counter.observe("q", 1, same), 0)
  assert.equal(counter.observe("q", 2, same), 1)
  assert.equal(counter.observe("q", 2, same), 1)
  assert.equal(counter.observe("q", 3, same), 2)
  assert.equal(counter.observe("q", 4, changed), 0)
  assert.equal(counter.observe("q", 5, changed), 1)

  counter.reset()
  assert.equal(counter.observe("q", 5, changed), 0)
  assert.equal(counter.observe("q", 6, changed), 0)
  assert.equal(counter.observe("q", 7, changed), 1)

  assert.equal(counter.observe("other", 7, changed), 0)
  assert.equal(responseFingerprint(undefined), null)
})

test("a query observer follows the waiting backoff and restarts after a reset", async (t) => {
  const timers = new ManualTimers()
  timeoutManager.setTimeoutProvider(timers)
  // Query Core disables refetch intervals outside a browser.
  environmentManager.setIsServer(() => false)
  t.after(() => {
    environmentManager.setIsServer(() => typeof window === "undefined")
    // Restoring the global timers makes Query Core warn about the provider switch.
    t.mock.method(console, "error", () => {})
    timeoutManager.setTimeoutProvider({
      setTimeout: (callback, delay) => setTimeout(callback, delay),
      clearTimeout: (id) => clearTimeout(id),
      setInterval: (callback, delay) => setInterval(callback, delay),
      clearInterval: (id) => clearInterval(id),
    })
  })
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const counter = new UnchangedResponseCounter()
  let calls = 0
  let response = { actions: [{ id: "approval-1" }], issues: [] }
  const observer = new QueryObserver(client, {
    queryKey: ["pending-actions", "session", "waiting"],
    queryFn: async () => {
      calls += 1
      return response
    },
    refetchInterval: (query) =>
      waitingPollInterval(
        counter.observe(
          query.queryHash,
          query.state.dataUpdateCount,
          responseFingerprint(query.state.data),
        ),
      ),
  })
  const unsubscribe = observer.subscribe(() => {})

  async function advance(ms) {
    timers.advance(ms)
    await settle()
  }

  try {
    await settle()
    assert.equal(calls, 1)
    for (const interval of [5000, 10_000, 30_000, 60_000, 60_000]) {
      const before = calls
      await advance(interval - 1)
      assert.equal(calls, before, `no request before ${interval} ms`)
      await advance(1)
      assert.equal(calls, before + 1, `one request after ${interval} ms`)
    }

    response = { actions: [{ id: "approval-2" }], issues: [] }
    await advance(60_000)
    const afterChange = calls
    await advance(5000)
    assert.equal(calls, afterChange + 1, "a changed response restarts at five seconds")

    await advance(10_000)
    const beforeReset = calls
    counter.reset()
    await observer.refetch()
    await settle()
    assert.equal(calls, beforeReset + 1)
    await advance(4999)
    assert.equal(calls, beforeReset + 1)
    await advance(1)
    assert.equal(calls, beforeReset + 2, "a reset restarts the backoff at five seconds")
  } finally {
    unsubscribe()
    client.clear()
  }
})
