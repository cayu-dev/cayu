import assert from "node:assert/strict"
import test from "node:test"
import {
  loadSessionIndexPage,
  MAX_AUTOMATIC_SESSION_PAGES,
} from "../src/lib/session-index-pages.ts"

const empty = (next_cursor) => ({ sessions: [], next_cursor, total_count: null })
test("empty continuations retain filters until a match or exhaustion", async () => {
  const calls = []
  const signal = new AbortController().signal
  const result = await loadSessionIndexPage(
    { q: "needle", limit: 100 },
    async (query, received) => {
      calls.push(query)
      assert.equal(received, signal)
      return query.cursor ? { ...empty(null), sessions: [{ id: "needle" }] } : empty("next")
    },
    signal,
  )
  assert.equal(result.page.sessions[0].id, "needle")
  assert.equal(result.searchPaused, false)
  assert.deepEqual(calls, [
    { q: "needle", limit: 100 },
    { q: "needle", limit: 100, cursor: "next" },
  ])
  const exhausted = await loadSessionIndexPage({}, async () => empty(null), signal)
  assert.equal(exhausted.searchPaused, false)
})
test("bounded automatic work pauses with an actionable cursor", async () => {
  let calls = 0
  const result = await loadSessionIndexPage(
    {},
    async () => empty(`page-${++calls}`),
    new AbortController().signal,
  )
  assert.equal(calls, MAX_AUTOMATIC_SESSION_PAGES)
  assert.equal(result.searchPaused, true)
  assert.equal(result.page.next_cursor, `page-${calls}`)
})
test("repeated cursor fails without retries; read errors stay authoritative", async () => {
  let calls = 0
  await assert.rejects(
    loadSessionIndexPage(
      { cursor: "same" },
      async () => {
        calls += 1
        return empty("same")
      },
      new AbortController().signal,
    ),
    (error) => error.status === 409 && /did not advance/.test(error.message),
  )
  assert.equal(calls, 1)
  const failure = new Error("service unavailable")
  await assert.rejects(
    loadSessionIndexPage(
      {},
      async (query) => {
        if (query.cursor) throw failure
        return empty("next")
      },
      new AbortController().signal,
    ),
    (error) => error === failure,
  )
})
test("abort fences late results even if the adapter ignores its signal", async () => {
  const controller = new AbortController()
  let release
  let calls = 0
  const result = loadSessionIndexPage(
    {},
    async () => {
      calls += 1
      return await new Promise((resolve) => {
        release = resolve
      })
    },
    controller.signal,
  )
  controller.abort()
  release(empty("must-not-follow"))
  await assert.rejects(result, (error) => error.name === "AbortError")
  assert.equal(calls, 1)
})

test("cursor cycles and pre-aborted work cannot dispatch indefinitely", async () => {
  let calls = 0
  await assert.rejects(
    loadSessionIndexPage(
      {},
      async () => empty(["a", "b", "a"][calls++]),
      new AbortController().signal,
    ),
    (error) => error.status === 409,
  )
  assert.equal(calls, 3)
  const controller = new AbortController()
  controller.abort()
  await assert.rejects(
    loadSessionIndexPage(
      {},
      async () => {
        assert.fail("pre-aborted work dispatched")
      },
      controller.signal,
    ),
    (error) => error.name === "AbortError",
  )
})

test("canonical overview envelopes retain aligned facts and nullable totals", async () => {
  const page = {
    sessions: [{ id: "visible" }],
    overviews: [{ session_id: "visible", state: "needs_recovery", tool_calls: "9007199254740993" }],
    next_cursor: "following",
    total_count: null,
  }
  const result = await loadSessionIndexPage(
    { include: ["overview"], include_forks: false },
    async () => page,
    new AbortController().signal,
  )
  assert.equal(result.page, page)
  assert.equal(result.page.overviews[0].tool_calls, "9007199254740993")
  assert.equal(result.page.total_count, null)
})
