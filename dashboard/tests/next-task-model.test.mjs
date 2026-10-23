import assert from "node:assert/strict"
import test from "node:test"
import { groupOf, holdable, isCayuTask, statusLine, when } from "../src/next/app/task-model.ts"

const NOW = Date.parse("2026-10-21T23:30:00Z") // 4:30 PM Pacific
const min = 60_000
const iso = (ms) => new Date(ms).toISOString()
const task = (over = {}) => ({
  id: "t1",
  type: "close_books",
  title: "Close October books",
  status: "pending",
  status_reason: null,
  session_id: null,
  parent_task_id: null,
  available_at: null,
  retry_series: null,
  schedule: null,
  prerequisite_task_ids: [],
  ...over,
})

test("statuses fall into five groups", () => {
  assert.equal(groupOf("claimed"), "active")
  assert.equal(groupOf("waiting_dependencies"), "waiting")
  assert.equal(groupOf("needs_attention"), "held")
  assert.equal(groupOf("dependency_skipped"), "failed")
  assert.equal(groupOf("cancelled"), "done")
})

test("near times are relative, later ones a Pacific clock time", () => {
  assert.equal(when(NOW + 6 * min, NOW), "in 6m")
  assert.equal(when(NOW - 5_000, NOW), "just now")
  assert.equal(when(NOW + 2 * 60 * min, NOW), "6:30 PM")
  assert.equal(when(Date.parse("2026-11-01T14:00:00Z"), NOW), "Nov 1, 6:00 AM")
})

test("waiting on prerequisites counts the ones not completed", () => {
  const all = [
    task({ id: "a", status: "completed" }),
    task({ id: "b", status: "running" }),
    task({ id: "c", status: "pending" }),
  ]
  const t = task({ status: "waiting_dependencies", prerequisite_task_ids: ["a", "b", "c"] })
  assert.equal(statusLine(t, [...all, t], NOW), "Waiting on 2 tasks")
})

test("pending tasks say when they start", () => {
  assert.equal(statusLine(task(), [], NOW), "Queued")
  const later = task({ available_at: iso(Date.parse("2026-11-01T14:00:00Z")), schedule: {} })
  assert.equal(statusLine(later, [], NOW), "Scheduled for Nov 1, 6:00 AM")
  const retry = task({
    available_at: iso(NOW + 6 * min),
    retry_series: { attempt: 2, next_eligible_at: iso(NOW + 6 * min), policy: { max_attempts: 3 } },
  })
  assert.equal(statusLine(retry, [], NOW), "Retry 2 of 3 in 6m")
})

test("holds and failures carry their reason; a retried failure points at the retry", () => {
  const blocked = task({ status: "blocked", status_reason: "Vendor file is late." })
  assert.equal(statusLine(blocked, [], NOW), "Blocked: Vendor file is late.")
  const next = task({ id: "t2", retry_series: { attempt: 2, policy: { max_attempts: 3 } } })
  const failed = task({
    status: "failed",
    status_reason: "rate limited",
    retry_series: { attempt: 1, successor_task_id: "t2", policy: { max_attempts: 3 } },
  })
  assert.equal(statusLine(failed, [failed, next], NOW), "Failed · attempt 2 queued")
  assert.equal(
    statusLine(task({ status: "failed", status_reason: "boom" }), [], NOW),
    "Failed: boom",
  )
})

test("running tasks count their subtasks", () => {
  const parent = task({ status: "running" })
  const kids = [task({ id: "k1", parent_task_id: "t1" }), task({ id: "k2", parent_task_id: "t1" })]
  assert.equal(statusLine(parent, [parent, ...kids], NOW), "Running · 2 subtasks")
})

test("Cayu's own tasks and what can be held", () => {
  assert.equal(isCayuTask({ type: "cayu.dispatch" }), true)
  assert.equal(isCayuTask({ type: "close_books" }), false)
  assert.equal(holdable(task()), true)
  assert.equal(holdable(task({ status: "running", session_id: "s1" })), false)
  assert.equal(holdable(task({ status: "blocked" })), false)
  assert.equal(holdable(task({ status: "completed" })), false)
})

test("failed retries increment the predecessor only when the successor has no attempt", () => {
  for (const [priorAttempt, nextAttempt, expected] of [
    [1, null, 2],
    [3, null, 4],
    [undefined, null, 2],
    [3, 8, 8],
  ]) {
    const next = task({
      id: "retry",
      retry_series: nextAttempt === null ? null : { attempt: nextAttempt },
    })
    const prior = task({
      status: "failed",
      retry_series: { attempt: priorAttempt, successor_task_id: "retry" },
    })
    assert.equal(statusLine(prior, [prior, next], NOW), `Failed · attempt ${expected} queued`)
  }
  assert.equal(statusLine(task({ status: "failed", retry_series: null }), [], NOW), "Failed")
})
