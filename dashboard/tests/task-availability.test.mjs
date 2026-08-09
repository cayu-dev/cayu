import assert from "node:assert/strict"
import test from "node:test"

import { taskAvailabilityDescriptor } from "../src/lib/task-availability.ts"

test("task availability labels describe durable configuration without using client time", () => {
  assert.equal(
    taskAvailabilityDescriptor({
      status: "pending",
      session_id: null,
      available_at: null,
    }),
    "Immediate",
  )
  assert.equal(
    taskAvailabilityDescriptor({
      status: "pending",
      session_id: "session-1",
      available_at: null,
    }),
    "Session-bound",
  )

  for (const availableAt of ["2000-01-01T00:00:00Z", "2999-01-01T00:00:00Z"]) {
    assert.equal(
      taskAvailabilityDescriptor({
        status: "pending",
        session_id: null,
        available_at: availableAt,
      }),
      "Time-gated",
    )
  }
})

test("non-pending tasks do not receive a pending-availability label", () => {
  assert.equal(
    taskAvailabilityDescriptor({
      status: "claimed",
      session_id: null,
      available_at: "2999-01-01T00:00:00Z",
    }),
    null,
  )
})
