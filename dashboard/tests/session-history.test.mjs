import assert from "node:assert/strict"
import test from "node:test"

import {
  CoalescedAsyncRunner,
  durationDetail,
  durationEndTimestamp,
  eventScanCursor,
  historyModeLabel,
  initialPageNavigation,
  jumpToLatestPage,
  loadStableTranscriptTailPage,
  mergeLatestEventWindow,
  mergeLatestTranscriptWindow,
  mergeStoredAndLiveRecords,
  navigateToNewerPage,
  navigateToOlderPage,
  olderTranscriptPage,
  pendingTailCanReconcile,
  queryReadErrorIsFatal,
  sessionMetadataNeedsRefresh,
  sessionSummaryRevision,
  statePollInterval,
  summaryStateIsStable,
  tailActivityAction,
} from "../src/lib/session-history.ts"

function deferred() {
  let resolve
  const promise = new Promise((complete) => {
    resolve = complete
  })
  return { promise, resolve }
}

test("coalesced async runners serialize activity and retain one catch-up pass", async () => {
  const runner = new CoalescedAsyncRunner()
  const firstRelease = deferred()
  let calls = 0
  let active = 0
  let maxActive = 0

  const task = async () => {
    calls += 1
    active += 1
    maxActive = Math.max(maxActive, active)
    if (calls === 1) await firstRelease.promise
    active -= 1
    return true
  }

  const firstRequest = runner.request(task)
  const secondRequest = runner.request(task)
  const thirdRequest = runner.request(task)
  assert.equal(runner.active, true)
  assert.equal(calls, 1)

  firstRelease.resolve()
  await Promise.all([firstRequest, secondRequest, thirdRequest])

  assert.equal(calls, 2)
  assert.equal(maxActive, 1)
  assert.equal(runner.active, false)
})

test("coalesced async runners discard queued work after cancellation or failure", async () => {
  for (const stop of ["clear", "failure"]) {
    const runner = new CoalescedAsyncRunner()
    const firstRelease = deferred()
    let calls = 0
    const task = async () => {
      calls += 1
      if (calls === 1) await firstRelease.promise
      return stop !== "failure"
    }

    const firstRequest = runner.request(task)
    runner.request(task)
    if (stop === "clear") runner.clearPending()
    firstRelease.resolve()
    await firstRequest

    assert.equal(calls, 1)
    assert.equal(runner.active, false)
  }
})

test("coalesced async runners reset after an unexpected task error", async () => {
  const runner = new CoalescedAsyncRunner()
  const firstRelease = deferred()
  let calls = 0
  const failingTask = async () => {
    calls += 1
    await firstRelease.promise
    throw new Error("read failed")
  }

  const firstRequest = runner.request(failingTask)
  runner.request(failingTask)
  firstRelease.resolve()
  await assert.rejects(firstRequest, /read failed/)

  assert.equal(calls, 1)
  assert.equal(runner.active, false)
  await runner.request(async () => {
    calls += 1
    return true
  })
  assert.equal(calls, 2)
})

test("page navigation keeps only cursors while moving older and newer", () => {
  let navigation = initialPageNavigation(null)
  navigation = navigateToOlderPage(navigation, 200)
  navigation = navigateToOlderPage(navigation, 100)

  assert.deepEqual(navigation, { current: 100, newer: [null, 200] })
  navigation = navigateToNewerPage(navigation)
  assert.deepEqual(navigation, { current: 200, newer: [null] })
  navigation = navigateToNewerPage(navigation)
  assert.deepEqual(navigation, { current: null, newer: [] })
  assert.deepEqual(navigateToNewerPage(navigation), navigation)
  assert.deepEqual(jumpToLatestPage(null), { current: null, newer: [] })
})

test("transcript navigation creates non-overlapping partial oldest pages", () => {
  assert.deepEqual(olderTranscriptPage(235, 100), { offset: 135, limit: 100 })
  assert.deepEqual(olderTranscriptPage(135, 100), { offset: 35, limit: 100 })
  assert.deepEqual(olderTranscriptPage(35, 100), { offset: 0, limit: 35 })
  assert.equal(olderTranscriptPage(0, 100), undefined)
})

test("summary revisions ignore hot activity but version stable terminal state", () => {
  const running = {
    status: "running",
    lastActivityAt: "2026-01-01T00:00:01Z",
    interruptionCascade: "none",
  }
  assert.equal(sessionSummaryRevision(running), "running:none:snapshot")
  assert.equal(
    sessionSummaryRevision({ ...running, lastActivityAt: "2026-01-01T00:00:02Z" }),
    "running:none:snapshot",
  )
  assert.equal(summaryStateIsStable(running), false)

  const completed = { ...running, status: "completed" }
  assert.equal(sessionSummaryRevision(completed), "completed:none:2026-01-01T00:00:01Z")
  assert.equal(summaryStateIsStable(completed), true)

  const cascading = {
    ...completed,
    status: "interrupted",
    interruptionCascade: "pending",
  }
  assert.equal(sessionSummaryRevision(cascading), "interrupted:pending:snapshot")
  assert.equal(summaryStateIsStable(cascading), false)

  const failedCascade = {
    ...cascading,
    lastActivityAt: "2026-01-01T00:00:03Z",
    interruptionCascade: "failed",
  }
  assert.equal(sessionSummaryRevision(failedCascade), "interrupted:failed:2026-01-01T00:00:03Z")
  assert.equal(summaryStateIsStable(failedCascade), true)

  const completedCascade = {
    ...failedCascade,
    lastActivityAt: "2026-01-01T00:00:04Z",
    interruptionCascade: "none",
  }
  assert.equal(sessionSummaryRevision(completedCascade), "interrupted:none:2026-01-01T00:00:04Z")
  assert.equal(summaryStateIsStable(completedCascade), true)
})

test("state polling keeps a low-frequency terminal heartbeat and backs off failures", () => {
  const base = {
    status: "running",
    interruptionCascade: "none",
    interruptRequested: false,
    mutationActive: false,
    hasError: false,
    failureCount: 0,
    nonRetryableError: false,
  }

  assert.equal(statePollInterval(base), 2000)
  assert.equal(statePollInterval({ ...base, interruptionCascade: "pending" }), 1000)
  assert.equal(statePollInterval({ ...base, status: "completed" }), 15_000)
  assert.equal(statePollInterval({ ...base, status: "interrupted", mutationActive: true }), 2000)
  assert.equal(statePollInterval({ ...base, hasError: true, failureCount: 1 }), 4000)
  assert.equal(statePollInterval({ ...base, hasError: true, failureCount: 20 }), 30_000)
  assert.equal(statePollInterval({ ...base, hasError: true, nonRetryableError: true }), false)
})

test("a state refetch error is nonfatal while prior state remains available", () => {
  assert.equal(queryReadErrorIsFatal(true, false), true)
  assert.equal(queryReadErrorIsFatal(true, true), false)
  assert.equal(queryReadErrorIsFatal(false, false), false)
})

test("session metadata refreshes only when the lightweight state revision advances", () => {
  assert.equal(sessionMetadataNeedsRefresh(undefined, undefined), false)
  assert.equal(sessionMetadataNeedsRefresh("2026-01-01T00:00:00Z", undefined), false)
  assert.equal(sessionMetadataNeedsRefresh("2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"), false)
  assert.equal(sessionMetadataNeedsRefresh("2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z"), true)
})

test("latest event windows deduplicate, order, and retain only the newest records", () => {
  const current = [
    { id: "event-3", sequence: 3 },
    { id: "event-2", sequence: 2 },
  ]
  const merged = mergeLatestEventWindow(
    current,
    [
      { id: "event-3", sequence: 3 },
      { id: "event-4", sequence: 4 },
      { id: "event-5", sequence: 5 },
    ],
    3,
  )

  assert.deepEqual(
    merged.records.map((event) => event.sequence),
    [5, 4, 3],
  )
  assert.equal(merged.dropped, true)
})

test("event tail cursors advance across filtered records without moving backwards", () => {
  assert.equal(eventScanCursor({ events: [], scan_through_sequence: 100 }, 5), 100)
  assert.equal(eventScanCursor({ events: [{ sequence: 105 }], scan_through_sequence: 100 }, 5), 105)
  assert.equal(eventScanCursor({ events: [], scan_through_sequence: 90 }, 100), 100)
})

test("durable events replace live copies without changing stable event ids", () => {
  const stored = [{ id: "stored-1" }, { id: "shared" }]
  const live = [{ id: "shared" }, { id: "live-1" }]

  assert.deepEqual(mergeStoredAndLiveRecords(stored, live), [
    { id: "stored-1" },
    { id: "shared" },
    { id: "live-1" },
  ])
})

test("latest transcript windows retain stable indexes without duplicates", () => {
  const merged = mergeLatestTranscriptWindow(
    [{ index: 10 }, { index: 11 }, { index: 12 }],
    [{ index: 12 }, { index: 13 }, { index: 14 }],
    4,
  )

  assert.deepEqual(
    merged.records.map((message) => message.index),
    [11, 12, 13, 14],
  )
  assert.equal(merged.dropped, true)
})

test("transcript tail reads follow bounded concurrent growth", async () => {
  const offsets = []
  const pages = [
    { has_more: true, total_messages: 236, marker: "behind" },
    { has_more: false, total_messages: 236, marker: "tail" },
  ]
  const page = await loadStableTranscriptTailPage(235, 100, 3, async (offset) => {
    offsets.push(offset)
    const next = pages.shift()
    assert.ok(next)
    return next
  })

  assert.deepEqual(offsets, [135, 136])
  assert.equal(page.marker, "tail")
})

test("transcript tail reads fail clearly instead of chasing writes forever", async () => {
  const offsets = []
  await assert.rejects(
    loadStableTranscriptTailPage(235, 100, 3, async (offset) => {
      offsets.push(offset)
      return { has_more: true, total_messages: offset + 101 }
    }),
    /growing too quickly/,
  )
  assert.deepEqual(offsets, [135, 136, 137])
})

test("history labels and duration descriptions report the actual UI state", () => {
  assert.equal(historyModeLabel({ atLatest: false, active: true, pinned: true }), "History")
  assert.equal(historyModeLabel({ atLatest: true, active: false, pinned: true }), "Latest")
  assert.equal(historyModeLabel({ atLatest: true, active: true, pinned: true }), "Live tail")
  assert.equal(historyModeLabel({ atLatest: true, active: true, pinned: false }), "Paused")
  assert.equal(historyModeLabel({ atLatest: true, active: false, pinned: false }), "Paused")

  assert.equal(durationDetail("completed", true), "started to completion")
  assert.equal(durationDetail("failed", true), "started to failure")
  assert.equal(durationDetail("interrupted", true), "started to interruption")
  assert.equal(durationDetail("completed", false), "started to last activity")
})

test("tail reconciliation freezes paused views and resumes after blockers clear", () => {
  assert.equal(tailActivityAction({ atLatest: true, pinned: true, visible: true }), "reconcile")
  assert.equal(tailActivityAction({ atLatest: true, pinned: false, visible: true }), "defer")
  assert.equal(tailActivityAction({ atLatest: true, pinned: true, visible: false }), "defer")
  assert.equal(tailActivityAction({ atLatest: false, pinned: true, visible: true }), "ignore")

  const pending = {
    pending: true,
    atLatest: true,
    pinned: true,
    visible: true,
    mutationActive: false,
    queryFetching: true,
    hasError: false,
  }
  assert.equal(pendingTailCanReconcile(pending), false)
  assert.equal(pendingTailCanReconcile({ ...pending, queryFetching: false }), true)
  assert.equal(pendingTailCanReconcile({ ...pending, mutationActive: true }), false)
  assert.equal(pendingTailCanReconcile({ ...pending, hasError: true }), false)
})

test("terminal durations prefer the durable terminal event over later activity", () => {
  assert.equal(
    durationEndTimestamp({
      terminalEventAt: "2026-01-01T00:00:10Z",
      lastActivityAt: "2026-01-01T00:05:00Z",
      updatedAt: "2026-01-01T00:05:00Z",
    }),
    "2026-01-01T00:00:10Z",
  )
  assert.equal(
    durationEndTimestamp({
      terminalEventAt: null,
      lastActivityAt: "2026-01-01T00:05:00Z",
      updatedAt: "2026-01-01T00:04:00Z",
    }),
    "2026-01-01T00:05:00Z",
  )
})
