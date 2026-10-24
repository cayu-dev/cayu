import assert from "node:assert/strict"
import test from "node:test"
import * as history from "../src/lib/session-history.ts"

const page = (sequences, through, more = false) => ({
  events: sequences.map((sequence) => ({ id: `event-${sequence}`, sequence })),
  scan_through_sequence: through,
  has_more: more,
})

test("shared progress follows scan cursors, drains backlog and reports a terminal tail", async () => {
  assert.equal(typeof history.followSessionProgress, "function")
  const abort = new AbortController()
  const requests = []
  const updates = []
  let active = true
  const pages = [page([10], 10), page([], 15, true), page([16], 16), page([17], 17)]
  await history.followSessionProgress({
    signal: abort.signal,
    active: () => active,
    read: async (query, signal) => {
      assert.equal(signal, abort.signal)
      requests.push(query)
      if (requests.length === 4) active = false
      return pages.shift()
    },
    onProgress: async (update) => {
      updates.push(update)
      if (update.throughSequence === 17) abort.abort()
    },
    sleep: async () => {},
  })
  assert.equal(requests[0].order_by, "sequence_desc")
  assert.equal(requests[0].limit, 1)
  assert.deepEqual(
    requests.slice(1).map((q) => q.after_sequence),
    [10, 15, 16],
  )
  assert.equal(requests[1].wait_seconds, 10)
  assert.equal(requests[2].wait_seconds, 0)
  assert.equal(updates.at(-1).terminalDrain, true)
  assert.deepEqual(
    updates.map((u) => u.throughSequence),
    [10, 15, 16, 17],
  )
})

test("a dispatched read cannot notify after abort even when adapter ignores the signal", async () => {
  assert.equal(typeof history.followSessionProgress, "function")
  const abort = new AbortController()
  let release
  let dispatched
  const entered = new Promise((resolve) => {
    dispatched = resolve
  })
  const updates = []
  const running = history.followSessionProgress({
    signal: abort.signal,
    active: () => true,
    read: async () => {
      dispatched()
      await new Promise((resolve) => {
        release = resolve
      })
      return page([100], 100)
    },
    onProgress: async (update) => {
      updates.push(update)
    },
    sleep: async () => {},
  })
  await entered
  abort.abort()
  release()
  await running
  assert.deepEqual(updates, [])
})

test("empty terminal reads back off and cancelled sleep releases the follower", async () => {
  assert.equal(typeof history.followSessionProgress, "function")
  const abort = new AbortController()
  const delays = []
  let calls = 0
  await history.followSessionProgress({
    signal: abort.signal,
    active: () => false,
    read: async () => {
      calls += 1
      return page([], 0)
    },
    onProgress: async () => {},
    sleep: async (ms) => {
      delays.push(ms)
      if (delays.length === 3) abort.abort()
    },
  })
  assert.equal(calls, 3)
  assert.deepEqual(delays, [5000, 10000, 30000])
})

test("terminal initial probes ignore older history and abort wakes a real delay", async () => {
  const abort = new AbortController()
  const updates = []
  const running = history.followSessionProgress({
    signal: abort.signal,
    active: () => false,
    read: async () => page([20], 20, true),
    onProgress: async (update) => {
      updates.push(update)
    },
  })
  await new Promise((resolve) => setTimeout(resolve, 5))
  abort.abort()
  await running
  assert.equal(updates.length, 1)
  assert.equal(updates[0].terminalDrain, true)
})
