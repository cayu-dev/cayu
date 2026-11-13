import assert from "node:assert/strict"
import test from "node:test"
import { reviewApiFromCayuClient } from "../src/parts/review-api.ts"

test("protected review reads through client.js with the purpose, reference and signal", async () => {
  const calls = []
  const view = {
    session_id: "s1",
    status: "permitted",
    guidance: "Review all calls before deciding.",
  }
  const field = { session_id: "s1", key: "body", status: "permitted", text: "All of it." }
  const api = reviewApiFromCayuClient({
    getHumanReview: async (...args) => {
      calls.push(["getHumanReview", ...args])
      return view
    },
    readHumanReviewLongField: async (...args) => {
      calls.push(["readHumanReviewLongField", ...args])
      return field
    },
  })
  const signal = new AbortController().signal
  const reference = {
    content_tag: "t",
    policy_version: "v1",
    context: { purpose: "p", recipient: "a" },
  }

  assert.equal(await api.inspect("s1", "p", signal), view)
  assert.equal(await api.longField("s1", reference, "body", signal), field)
  assert.deepEqual(calls, [
    ["getHumanReview", "s1", { purpose: "p", signal }],
    ["readHumanReviewLongField", "s1", reference, "body", { signal }],
  ])
})
