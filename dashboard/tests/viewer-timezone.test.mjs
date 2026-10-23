import assert from "node:assert/strict"
import test from "node:test"
import { relativeTime } from "../src/next/app/session-model.ts"
import { when } from "../src/next/app/task-model.ts"

test("task clocks and session day boundaries use the viewer timezone by default", () => {
  const previous = process.env.TZ
  try {
    process.env.TZ = "Asia/Tokyo"
    const now = Date.parse("2026-10-23T18:00:00Z")
    const later = Date.parse("2026-10-23T21:00:00Z")
    assert.equal(when(later, now), when(later, now, "Asia/Tokyo"))
    const prior = Date.parse("2026-10-23T14:00:00Z")
    assert.equal(relativeTime(prior, now), "Yesterday")
    assert.equal(relativeTime(prior, now, "America/Los_Angeles"), "4h")
  } finally {
    if (previous === undefined) delete process.env.TZ
    else process.env.TZ = previous
  }
})
