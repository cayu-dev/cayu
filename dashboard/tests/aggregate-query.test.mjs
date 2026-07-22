import assert from "node:assert/strict"
import test from "node:test"
import {
  aggregateRequestFailureIsPermanent,
  aggregateSnapshotNeedsWarning,
  retryAggregateRequest,
} from "../src/lib/aggregate-query.ts"

class HttpError extends Error {
  constructor(status) {
    super(`HTTP ${status}`)
    this.status = status
  }
}

test("aggregate retries distinguish transient and permanent HTTP failures", () => {
  for (const status of [408, 425, 429, 500, 503]) {
    const error = new HttpError(status)
    assert.equal(aggregateRequestFailureIsPermanent(error), false)
    assert.equal(retryAggregateRequest(0, error), true)
    assert.equal(retryAggregateRequest(3, error), false)
  }

  for (const status of [400, 401, 403, 404, 422, 501]) {
    const error = new HttpError(status)
    assert.equal(aggregateRequestFailureIsPermanent(error), true)
    assert.equal(retryAggregateRequest(0, error), false)
  }
})

test("an intentionally absent task store does not make exact session metrics a warning", () => {
  const exactSessions = { accuracy: { kind: "exact" } }
  assert.equal(
    aggregateSnapshotNeedsWarning(
      { sessions: exactSessions, task_snapshot_status: "not_configured", tasks: null },
      false,
    ),
    false,
  )
  assert.equal(
    aggregateSnapshotNeedsWarning(
      { sessions: exactSessions, task_snapshot_status: "unsupported", tasks: null },
      false,
    ),
    true,
  )
  assert.equal(
    aggregateSnapshotNeedsWarning(
      {
        sessions: exactSessions,
        task_snapshot_status: "available",
        tasks: { accuracy: { kind: "sampled" } },
      },
      false,
    ),
    true,
  )
})
