import assert from "node:assert/strict"
import test from "node:test"
import { startBrowserPageObserver } from "../src/lib/browser-page-observer.ts"

async function settle() {
  await Promise.resolve()
  await Promise.resolve()
}

test("origin reads are serialized and stopping discards a pending result", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] })
  let release
  let reads = 0
  const published = []
  const owner = startBrowserPageObserver(
    () => {
      reads++
      return new Promise((resolve) => {
        release = resolve
      })
    },
    (value) => published.push(value),
    () => assert.fail("No read failure"),
    () => true,
  )
  t.mock.timers.tick(2_000)
  assert.equal(reads, 1)
  t.mock.timers.tick(20_000)
  assert.equal(reads, 1)
  owner.stop()
  release("late-origin")
  await settle()
  t.mock.timers.tick(20_000)
  assert.equal(reads, 1)
  assert.deepEqual(published, [])
})

test("busy reads cannot replace observations and failures stop automatic refresh", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] })
  let allowed = false
  let reads = 0
  let failures = 0
  const published = []
  const owner = startBrowserPageObserver(
    async () => {
      reads++
      if (reads === 1) {
        allowed = false
        return "old-origin"
      }
      throw Error("private-canary")
    },
    (value) => published.push(value),
    () => failures++,
    () => allowed,
  )
  t.mock.timers.tick(2_000)
  assert.equal(reads, 0)
  allowed = true
  t.mock.timers.tick(2_000)
  await settle()
  assert.deepEqual(published, [])
  allowed = true
  t.mock.timers.tick(2_000)
  await settle()
  t.mock.timers.tick(20_000)
  assert.equal(reads, 2)
  assert.equal(failures, 1)
  owner.stop()
})

test("origin observation has a finite lifetime and publishes fresh reads", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] })
  let now = 0
  t.mock.method(performance, "now", () => now)
  let expired = 0
  const published = []
  const owner = startBrowserPageObserver(
    async () => "https://changed.test",
    (value) => published.push(value),
    () => expired++,
    () => true,
  )
  t.mock.timers.tick(2_000)
  await settle()
  assert.deepEqual(published, ["https://changed.test"])
  now = 300_000
  t.mock.timers.tick(2_000)
  assert.equal(expired, 1)
  t.mock.timers.tick(20_000)
  assert.equal(expired, 1)
  owner.stop()
})
