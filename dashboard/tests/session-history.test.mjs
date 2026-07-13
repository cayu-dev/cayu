import assert from "node:assert/strict"
import test from "node:test"

import {
  CoalescedAsyncRunner,
  durationDetail,
  durationEndTimestamp,
  eventHistoryFilterCount,
  eventHistoryFilterKey,
  eventHistoryFilters,
  eventHistoryQuery,
  eventMatchesHistoryFilters,
  eventScanCursor,
  hasEventHistoryFilters,
  hasTranscriptHistoryFilters,
  historyModeLabel,
  initialFilteredPageNavigation,
  initialPageNavigation,
  jumpToLatestPage,
  latestTranscriptPageOffset,
  loadStableTranscriptTailPage,
  mergeLatestEventWindow,
  mergeLatestTranscriptWindow,
  mergeStoredAndLiveRecords,
  navigateToNewerPage,
  navigateToOlderPage,
  olderTranscriptPage,
  pageNavigationForFilter,
  pendingTailCanReconcile,
  queryReadErrorIsFatal,
  sessionHistorySearchWithEventFilters,
  sessionHistorySearchWithTranscriptFilters,
  sessionMetadataNeedsRefresh,
  sessionSummaryRevision,
  statePollInterval,
  summaryStateIsStable,
  tailActivityAction,
  transcriptDeltaRequiresTailReload,
  transcriptHistoryFilterKey,
  transcriptHistoryFilters,
  transcriptHistoryQuery,
} from "../src/lib/session-history.ts"
import { validateSessionHistorySearch } from "../src/lib/session-history-search.ts"

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

test("history search validation canonicalizes supported shareable filters", () => {
  const search = validateSessionHistorySearch({
    event_id: "  event-1  ",
    event_type: "tool.call.completed",
    tool_name: ["invalid"],
    agent_name: "builder",
    environment_name: "  ",
    workflow_name: "release",
    transcript_role: "assistant",
    include_thinking: "false",
    unrelated: "ignored",
  })

  assert.deepEqual(search, {
    event_id: "event-1",
    event_type: "tool.call.completed",
    tool_name: undefined,
    agent_name: "builder",
    environment_name: undefined,
    workflow_name: "release",
    transcript_role: "assistant",
    include_thinking: false,
  })
  assert.deepEqual(validateSessionHistorySearch({ transcript_role: "invalid" }), {
    event_id: undefined,
    event_type: undefined,
    tool_name: undefined,
    agent_name: undefined,
    environment_name: undefined,
    workflow_name: undefined,
    transcript_role: undefined,
    include_thinking: undefined,
  })
})

test("event filters round-trip through URL state without replacing transcript filters", () => {
  const current = validateSessionHistorySearch({
    event_type: "model.completed",
    transcript_role: "tool",
    include_thinking: false,
  })
  const filters = eventHistoryFilters(current)
  assert.equal(hasEventHistoryFilters(filters), true)
  assert.equal(eventHistoryFilterCount(filters), 1)

  const next = sessionHistorySearchWithEventFilters(current, {
    eventId: " event-9 ",
    eventType: "",
    toolName: "write_file",
    agentName: "",
    environmentName: "prod",
    workflowName: "",
  })
  assert.deepEqual(next, {
    event_id: "event-9",
    event_type: undefined,
    tool_name: "write_file",
    agent_name: undefined,
    environment_name: "prod",
    workflow_name: undefined,
    transcript_role: "tool",
    include_thinking: false,
  })
  assert.equal(
    eventHistoryFilterKey(eventHistoryFilters(next)),
    eventHistoryFilterKey({
      eventId: "event-9",
      eventType: "",
      toolName: "write_file",
      agentName: "",
      environmentName: "prod",
      workflowName: "",
    }),
  )
})

test("event queries hide deltas by default but honor explicit identity and type lookups", () => {
  const empty = {
    eventId: "",
    eventType: "",
    toolName: "",
    agentName: "",
    environmentName: "",
    workflowName: "",
  }
  assert.deepEqual(eventHistoryQuery(empty), { exclude_event_type: "model.text.delta" })
  assert.deepEqual(eventHistoryQuery({ ...empty, toolName: "read_file" }), {
    tool_name: "read_file",
    exclude_event_type: "model.text.delta",
  })
  assert.deepEqual(eventHistoryQuery({ ...empty, eventId: "delta-1" }), {
    event_id: "delta-1",
  })
  assert.deepEqual(eventHistoryQuery({ ...empty, eventType: "model.text.delta" }), {
    event_type: "model.text.delta",
  })
})

test("live events use the same exact filter semantics as stored event queries", () => {
  const event = {
    id: "event-1",
    type: "tool.call.completed",
    tool_name: "write_file",
    agent_name: "builder",
    environment_name: "prod",
    workflow_name: "release",
  }
  const filters = {
    eventId: "event-1",
    eventType: "tool.call.completed",
    toolName: "write_file",
    agentName: "builder",
    environmentName: "prod",
    workflowName: "release",
  }
  assert.equal(eventMatchesHistoryFilters(event, filters), true)
  assert.equal(eventMatchesHistoryFilters(event, { ...filters, workflowName: "other" }), false)
  assert.equal(
    eventMatchesHistoryFilters(
      { ...event, id: "delta", type: "model.text.delta" },
      { ...filters, eventId: "", eventType: "", toolName: "" },
    ),
    false,
  )
  assert.equal(
    eventMatchesHistoryFilters(
      { ...event, id: "delta", type: "model.text.delta" },
      { ...filters, eventId: "delta", eventType: "model.text.delta", toolName: "" },
    ),
    true,
  )
})

test("filter generations expose a fresh latest page before state reset effects run", () => {
  const state = initialFilteredPageNavigation("old-filter", null)
  state.navigation = navigateToOlderPage(state.navigation, 200)

  assert.deepEqual(pageNavigationForFilter(state, "old-filter", null), {
    current: 200,
    newer: [null],
  })
  assert.deepEqual(pageNavigationForFilter(state, "new-filter", null), {
    current: null,
    newer: [],
  })
})

test("transcript filters preserve event URL state and use role-relative query offsets", () => {
  const current = validateSessionHistorySearch({ event_id: "event-1" })
  const next = sessionHistorySearchWithTranscriptFilters(current, {
    role: "assistant",
    includeThinking: false,
  })
  const filters = transcriptHistoryFilters(next)

  assert.deepEqual(next, {
    event_id: "event-1",
    event_type: undefined,
    tool_name: undefined,
    agent_name: undefined,
    environment_name: undefined,
    workflow_name: undefined,
    transcript_role: "assistant",
    include_thinking: false,
  })
  assert.deepEqual(filters, { role: "assistant", includeThinking: false })
  assert.deepEqual(transcriptHistoryQuery(filters), {
    role: "assistant",
    include_thinking: false,
  })
  assert.equal(hasTranscriptHistoryFilters(filters), true)
  assert.equal(transcriptHistoryFilterKey(filters), '["assistant",false]')
  assert.equal(latestTranscriptPageOffset(235, 100), 135)
})

test("thinking-filtered transcript tails reload only when a bounded merge is ambiguous", () => {
  assert.equal(
    transcriptDeltaRequiresTailReload(
      { offset: 0, total_messages: 100, messages: Array.from({ length: 100 }) },
      { total_messages: 103, messages: [{ index: 100 }, { index: 102 }] },
      false,
      100,
    ),
    true,
  )
  assert.equal(
    transcriptDeltaRequiresTailReload(
      { offset: 100, total_messages: 200, messages: Array.from({ length: 75 }) },
      { total_messages: 201, messages: [{}] },
      false,
      100,
    ),
    true,
  )
  assert.equal(
    transcriptDeltaRequiresTailReload(
      { offset: 0, total_messages: 90, messages: Array.from({ length: 70 }) },
      { total_messages: 95, messages: [{}, {}, {}, {}, {}] },
      false,
      100,
    ),
    false,
  )
  assert.equal(
    transcriptDeltaRequiresTailReload(
      { offset: 0, total_messages: 100, messages: Array.from({ length: 90 }) },
      { total_messages: 103, messages: [] },
      true,
      100,
    ),
    false,
  )
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
