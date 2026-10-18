import assert from "node:assert/strict"
import test from "node:test"

import { sessionDeadlineState, sessionExecutionLabel } from "../src/lib/session-execution.ts"

test("execution presence distinguishes local and remote workers", () => {
  assert.equal(sessionExecutionLabel({ state: "executing", local_owner: true }), "Running here")
  assert.equal(
    sessionExecutionLabel({ state: "executing", local_owner: false }),
    "Running in another worker",
  )
})

test("expired ownership stays visible beside durable running or interrupting status", () => {
  assert.equal(sessionExecutionLabel({ state: "owner_lost" }), "Execution owner lost")
  assert.equal(sessionExecutionLabel({ state: "unknown" }), "Execution owner unknown")
})

test("human waits and settled execution do not claim a living owner", () => {
  assert.equal(sessionExecutionLabel({ state: "waiting" }), "Waiting")
  assert.equal(sessionExecutionLabel({ state: "idle" }), "Idle")
  assert.equal(sessionExecutionLabel({ state: "terminal" }), "Execution finished")
})

test("execution deadlines show expiry and use durable timeout evidence", () => {
  const metadata = { "cayu:execution_deadline": { expires_at: "2026-01-01T00:00:00Z" } }
  assert.deepEqual(
    sessionDeadlineState(metadata, {
      type: "session.interrupted",
      payload: { failure_evidence: { classification: "deadline" } },
    }),
    { expiresAt: "2026-01-01T00:00:00Z", timedOut: true },
  )
  assert.equal(
    sessionDeadlineState(metadata, {
      type: "session.failed",
      payload: { error_type: "ExecutionDeadlineExceeded" },
    }).timedOut,
    true,
  )
  // A past expiry alone does not turn an unrelated interruption into a timeout.
  assert.equal(
    sessionDeadlineState(metadata, {
      type: "session.interrupted",
      payload: { failure_evidence: { classification: "interruption" } },
    }).timedOut,
    false,
  )
  assert.equal(sessionDeadlineState(metadata).timedOut, false)
})

test("unbounded and malformed deadline metadata has no displayed expiry", () => {
  for (const value of [undefined, null, [], {}, { expires_at: null }, { expires_at: "invalid" }]) {
    assert.deepEqual(sessionDeadlineState({ "cayu:execution_deadline": value }), {
      expiresAt: null,
      timedOut: false,
    })
  }
})
