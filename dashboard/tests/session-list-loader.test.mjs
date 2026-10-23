import assert from "node:assert/strict"
import test from "node:test"
import { loadSessionList } from "../src/next/app/session-list-loader.ts"

const session = (id, labels = {}) => ({ id, labels })
const overview = (id, title = null) => ({ session_id: id, title, state: "completed" })
const signal = () => new AbortController().signal

test("reads parallel authoritative overviews and only bounded title fallbacks for visible matches", async () => {
  const calls = []
  const result = await loadSessionList(
    { include_eval: false },
    async (query) =>
      query.cursor
        ? {
            sessions: [
              session("authored"),
              session("labels", { x: "y" }),
              session("message"),
              session("empty"),
            ],
            overviews: [
              overview("authored", "Report"),
              overview("labels"),
              overview("message"),
              overview("empty"),
            ],
            next_cursor: null,
            total_count: null,
          }
        : { sessions: [], overviews: [], next_cursor: "next", total_count: null },
    async (id, query) => {
      calls.push([id, query])
      return {
        messages:
          id === "message" ? [{ role: "user", content: [{ type: "text", text: "Question" }] }] : [],
      }
    },
    signal(),
  )
  assert.deepEqual(
    result.rows.map((row) => row.title),
    ["Report", "x=y", "Question", "empty"],
  )
  assert.deepEqual(calls, [
    ["message", { role: "user", limit: 1 }],
    ["empty", { role: "user", limit: 1 }],
  ])
  assert.equal(result.page.total_count, null)
})

test("missing overview fails; title-read errors stay explicit and cancellation publishes nothing", async () => {
  const page = {
    sessions: [session("a")],
    overviews: [overview("a")],
    total_count: null,
    next_cursor: null,
  }
  await assert.rejects(
    loadSessionList(
      {},
      async () => ({ ...page, overviews: [] }),
      async () => ({}),
      signal(),
    ),
    /overview/,
  )
  const unavailable = await loadSessionList(
    {},
    async () => page,
    async () => {
      throw new Error("offline")
    },
    signal(),
  )
  assert.equal(unavailable.rows[0].title, "a")
  assert.equal(unavailable.rows[0].titleUnavailable, true)
  const controller = new AbortController()
  await assert.rejects(
    loadSessionList(
      {},
      async () => page,
      async () => {
        controller.abort()
        return { messages: [] }
      },
      controller.signal,
    ),
    { name: "AbortError" },
  )
})
