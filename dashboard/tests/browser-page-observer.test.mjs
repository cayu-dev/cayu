import assert from "node:assert/strict"
import { test, vi } from "vite-plus/test"
import { useFakeTimeouts } from "./fake-timeouts.mjs"
import { startBrowserPageObserver } from "../src/lib/browser-page-observer.ts"

async function settle() {
  await Promise.resolve()
  await Promise.resolve()
}

test("origin reads are serialized and stopping discards a pending result", async () => {
  useFakeTimeouts()
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
  vi.advanceTimersByTime(2_000)
  assert.equal(reads, 1)
  vi.advanceTimersByTime(20_000)
  assert.equal(reads, 1)
  owner.stop()
  release("late-origin")
  await settle()
  vi.advanceTimersByTime(20_000)
  assert.equal(reads, 1)
  assert.deepEqual(published, [])
})

test("busy reads cannot replace observations and failures stop automatic refresh", async () => {
  useFakeTimeouts()
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
  vi.advanceTimersByTime(2_000)
  assert.equal(reads, 0)
  allowed = true
  vi.advanceTimersByTime(2_000)
  await settle()
  assert.deepEqual(published, [])
  allowed = true
  vi.advanceTimersByTime(2_000)
  await settle()
  vi.advanceTimersByTime(20_000)
  assert.equal(reads, 2)
  assert.equal(failures, 1)
  owner.stop()
})

test("origin observation has a finite lifetime and publishes fresh reads", async () => {
  useFakeTimeouts()
  let now = 0
  vi.spyOn(performance, "now").mockImplementation(() => now)
  let expired = 0
  const published = []
  const owner = startBrowserPageObserver(
    async () => "https://changed.test",
    (value) => published.push(value),
    () => expired++,
    () => true,
  )
  vi.advanceTimersByTime(2_000)
  await settle()
  assert.deepEqual(published, ["https://changed.test"])
  now = 300_000
  vi.advanceTimersByTime(2_000)
  assert.equal(expired, 1)
  vi.advanceTimersByTime(20_000)
  assert.equal(expired, 1)
  owner.stop()
})
