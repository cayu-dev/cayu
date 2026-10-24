import assert from "node:assert/strict"
import test from "node:test"
import { taskGroupCount, taskGroupStatuses } from "../src/next/app/task-index-model.ts"

test("status chips sum exact server counts beyond safe JS integer range", () => {
  const counts = { running: "9007199254740993", claimed: "2", paused: "3", pending: "7" }
  assert.equal(taskGroupCount(counts, "active"), "9007199254740995")
  assert.equal(taskGroupCount(counts), "9007199254741005")
  assert.deepEqual(taskGroupStatuses("held"), ["paused", "blocked", "needs_attention"])
})
