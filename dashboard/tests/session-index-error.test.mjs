import assert from "node:assert/strict"
import test from "node:test"

import { sessionIndexErrorState } from "../src/lib/session-index-error.ts"

class HttpError extends Error {
  constructor(status, message = `HTTP ${status}`) {
    super(message)
    this.status = status
  }
}

test("session-index errors distinguish validation and access failures", () => {
  assert.deepEqual(sessionIndexErrorState(new HttpError(422, "Invalid label selector.")), {
    kind: "validation",
    title: "The session filters are invalid.",
    detail: "Invalid label selector.",
    retryable: false,
  })
  assert.equal(sessionIndexErrorState(new HttpError(401)).kind, "authentication")
  assert.equal(sessionIndexErrorState(new HttpError(403)).kind, "authorization")
})

test("session-index errors make only transient failures retryable", () => {
  assert.deepEqual(sessionIndexErrorState(new TypeError("Failed to fetch")), {
    kind: "network",
    title: "The Cayu server could not be reached.",
    detail: "Check the connection and try again.",
    retryable: true,
  })
  assert.equal(sessionIndexErrorState(new HttpError(503)).kind, "server")
  assert.equal(sessionIndexErrorState(new HttpError(503)).retryable, true)
  assert.equal(sessionIndexErrorState(new HttpError(409)).kind, "request")
  assert.equal(sessionIndexErrorState(new HttpError(409)).retryable, false)
  assert.equal(sessionIndexErrorState(new Error("Malformed response")).kind, "unexpected")
})
