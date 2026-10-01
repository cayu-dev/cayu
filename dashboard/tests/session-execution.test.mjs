import assert from "node:assert/strict"
import test from "node:test"

import { sessionExecutionLabel } from "../src/lib/session-execution.ts"

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
