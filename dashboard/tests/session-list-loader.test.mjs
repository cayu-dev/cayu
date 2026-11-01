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

test("first_message from the overview replaces per-row transcript reads when the server has it", async () => {
  const queries = []
  const reads = []
  const page = {
    sessions: [session("message"), session("empty"), session("labels", { x: "y" })],
    overviews: [
      { ...overview("message"), first_message: "Question\nwith details" },
      { ...overview("empty"), first_message: null },
      { ...overview("labels"), first_message: "Ignored" },
    ],
    next_cursor: null,
    total_count: null,
  }
  const readPage = async (query) => {
    queries.push(query)
    return page
  }
  const readFirstUser = async (id, query) => {
    reads.push([id, query])
    return { messages: [] }
  }
  const result = await loadSessionList(
    { include: ["overview"], include_eval: false },
    readPage,
    readFirstUser,
    signal(),
    { firstMessage: true },
  )
  assert.deepEqual(queries[0].include, ["overview", "first_message"])
  assert.deepEqual(
    result.rows.map((row) => row.title),
    ["Question", "empty", "x=y"],
  )
  assert.equal(result.rows[0].titleUnavailable, false)
  assert.deepEqual(reads, [])

  // Older servers: the include is never sent and rows fall back to one bounded read each.
  queries.length = 0
  await loadSessionList({ include: ["overview"] }, readPage, readFirstUser, signal(), {
    firstMessage: false,
  })
  assert.deepEqual(queries[0].include, ["overview"])
  assert.deepEqual(reads, [
    ["message", { role: "user", limit: 1 }],
    ["empty", { role: "user", limit: 1 }],
  ])
})

test("overview preview failures stay local to untitled rows", async () => {
  const rows = [
    session("unavailable"),
    session("empty"),
    session("titled"),
    session("labelled", { x: "y" }),
  ]
  const result = await loadSessionList(
    { include: ["overview"] },
    async () => ({
      sessions: rows,
      overviews: rows.map((row) => ({
        ...overview(row.id),
        title: row.id === "titled" ? "Saved title" : null,
        first_message: null,
        first_message_unavailable: row.id !== "empty",
      })),
      next_cursor: null,
      total_count: null,
    }),
    async () => {
      throw new Error("No transcript fallback expected")
    },
    signal(),
    { firstMessage: true },
  )
  assert.deepEqual(
    result.rows.map((row) => row.titleUnavailable),
    [true, false, false, false],
  )
  assert.deepEqual(
    result.rows.map((row) => row.title),
    ["unavailable", "empty", "Saved title", "x=y"],
  )
})
