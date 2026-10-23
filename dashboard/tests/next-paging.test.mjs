import assert from "node:assert/strict"
import test from "node:test"
import {
  filterInboxItems,
  loadEventHistory,
  loadEventTail,
  mergeEvents,
  readAllInbox,
} from "../src/next/api/paging.ts"
import { isEvalTrialMetadata, shouldFollowRowClick } from "../src/next/app/session-model.ts"

const event = (sequence) => ({ sequence, id: `e${sequence}`, type: "x", payload: {} })
const empty = { events: [], truncated: false, through: null }

/** A fake events endpoint over sequences 1..n, paging like the server. */
function eventServer(n) {
  const all = Array.from({ length: n }, (_, i) => event(i + 1))
  const calls = []
  const read = async (q) => {
    calls.push(q)
    let rows = all
    if (q.after_sequence != null) rows = rows.filter((e) => e.sequence > q.after_sequence)
    if (q.before_sequence != null) rows = rows.filter((e) => e.sequence < q.before_sequence)
    if (q.order_by === "sequence_desc") rows = [...rows].reverse()
    const page = rows.slice(0, q.limit)
    return {
      events: page,
      has_more: rows.length > q.limit,
      scan_through_sequence: q.order_by === "sequence_asc" ? (page.at(-1)?.sequence ?? null) : null,
    }
  }
  return { all, calls, read }
}

test("mergeEvents dedupes by sequence, sorts, and keeps the newest up to the cap", () => {
  const first = mergeEvents(empty, [event(3), event(1), event(2)], 10)
  assert.deepEqual(
    first.events.map((e) => e.sequence),
    [1, 2, 3],
  )
  assert.equal(first.through, 3)
  assert.equal(first.truncated, false)
  const next = mergeEvents(first, [event(3), event(4), event(5)], 4)
  assert.deepEqual(
    next.events.map((e) => e.sequence),
    [2, 3, 4, 5],
  )
  assert.equal(next.truncated, true)
})

test("mergeEvents advances the tail cursor past scanned sequences", () => {
  assert.equal(mergeEvents({ ...empty, through: 7 }, [], 10, 12).through, 12)
  assert.equal(mergeEvents({ ...empty, through: 7 }, [], 10, null).through, 7)
})

test("loadEventHistory reads one page when everything fits", async () => {
  const server = eventServer(40)
  const history = await loadEventHistory(server.read, 100, 50)
  assert.equal(history.events.length, 40)
  assert.equal(history.truncated, false)
  assert.equal(server.calls.length, 1)
  assert.ok(server.calls.every((q) => q.limit <= 1000))
})

test("loadEventHistory pages back with before_sequence and stops at the cap", async () => {
  const server = eventServer(250)
  const history = await loadEventHistory(server.read, 120, 50)
  assert.equal(history.events.length, 120)
  assert.equal(history.truncated, true)
  // The newest are kept, so the tail can continue from them.
  assert.equal(history.events[0].sequence, 131)
  assert.equal(history.events.at(-1).sequence, 250)
  assert.equal(history.through, 250)
  assert.deepEqual(
    server.calls.map((q) => [q.limit, q.before_sequence]),
    [
      [50, undefined],
      [50, 201],
      [20, 151],
    ],
  )
})

test("loadEventHistory never asks for more than the server's page cap", async () => {
  const server = eventServer(2500)
  const history = await loadEventHistory(server.read)
  assert.equal(history.events.length, 2500)
  assert.equal(history.truncated, false)
  assert.ok(server.calls.every((q) => q.limit <= 1000))
  assert.equal(server.calls.length, 3)
})

test("loadEventTail reads only after the newest event and long-polls once", async () => {
  const server = eventServer(2300)
  const start = mergeEvents(empty, server.all.slice(0, 200), 10_000)
  const tail = await loadEventTail(server.read, start, { wait: 10 })
  assert.equal(tail.events.length, 2300)
  assert.equal(server.calls[0].after_sequence, 200)
  assert.equal(server.calls[0].wait_seconds, 10)
  assert.ok(server.calls.slice(1).every((q) => q.wait_seconds === undefined))
  assert.ok(server.calls.every((q) => q.order_by === "sequence_asc" && q.limit <= 1000))
})

test("loadEventTail with nothing new keeps the history", async () => {
  const server = eventServer(5)
  const start = mergeEvents(empty, server.all, 10)
  const tail = await loadEventTail(server.read, start)
  assert.equal(tail.events.length, 5)
  assert.equal(server.calls.length, 1)
  assert.equal(server.calls[0].wait_seconds, undefined)
})

const item = (id, extra = {}) => ({ id, kind: "approval", session_id: "s1", ...extra })

test("readAllInbox follows cursors across empty pages and keeps issues", async () => {
  const pages = {
    "": { items: [item("a")], next_cursor: "sealed_1", has_more: true, issues: [{ code: "x" }] },
    sealed_1: { items: [], next_cursor: "sealed_2", has_more: true },
    sealed_2: { items: [item("b"), item("a")], next_cursor: null, has_more: false },
  }
  const seen = []
  const page = await readAllInbox(async (cursor) => {
    seen.push(cursor)
    return pages[cursor ?? ""]
  })
  assert.deepEqual(seen, [null, "sealed_1", "sealed_2"])
  assert.deepEqual(
    page.items.map((i) => i.id),
    ["a", "b"],
  )
  assert.deepEqual(page.issues, [{ code: "x" }])
  assert.equal(page.next_cursor, null)
  assert.equal(page.has_more, false)
})

test("readAllInbox stops at the page cap and reports more", async () => {
  let n = 0
  const page = await readAllInbox(
    async () => ({ items: [item(`i${++n}`)], next_cursor: `c${n}`, has_more: true }),
    { maxPages: 3 },
  )
  assert.equal(page.items.length, 3)
  assert.equal(page.has_more, true)
  assert.equal(page.next_cursor, "c3")
})

test("readAllInbox restarts once from no cursor on a stale cursor", async () => {
  const stale = Object.assign(new Error("422"), { status: 422 })
  let attempt = 0
  const seen = []
  const page = await readAllInbox(
    async (cursor) => {
      seen.push(cursor)
      if (cursor === null) {
        attempt++
        return { items: [item(`first${attempt}`)], next_cursor: "c", has_more: true }
      }
      if (attempt === 1) throw stale
      return { items: [item("second")], next_cursor: null, has_more: false }
    },
    { isStaleCursor: (e) => e.status === 422 },
  )
  assert.deepEqual(seen, [null, "c", null, "c"])
  // The pages read before the restart are dropped.
  assert.deepEqual(
    page.items.map((i) => i.id),
    ["first2", "second"],
  )
})

test("readAllInbox gives up after one restart and on errors without a cursor", async () => {
  const stale = Object.assign(new Error("422"), { status: 422 })
  const isStaleCursor = (e) => e.status === 422
  await assert.rejects(
    readAllInbox(
      async (cursor) => {
        if (cursor) throw stale
        return { items: [], next_cursor: "c", has_more: true }
      },
      { isStaleCursor },
    ),
    /422/,
  )
  let calls = 0
  await assert.rejects(
    readAllInbox(
      async () => {
        calls++
        throw stale
      },
      { isStaleCursor },
    ),
    /422/,
  )
  assert.equal(calls, 1)
})

test("filterInboxItems keeps only the session, agent and kind asked for", () => {
  const items = [
    item("a", { session_id: "s1", agent_name: "x", kind: "approval" }),
    item("b", { session_id: "s2", agent_name: "x", kind: "question" }),
    item("c", { session_id: "s1", agent_name: "y", kind: "question" }),
  ]
  const ids = (q) => filterInboxItems(items, q).map((i) => i.id)
  assert.deepEqual(ids({}), ["a", "b", "c"])
  assert.deepEqual(ids({ session_id: "s1" }), ["a", "c"])
  assert.deepEqual(ids({ agent_name: "x", kind: "question" }), ["b"])
})

test("isEvalTrialMetadata counts only a true cayu:eval_trial", () => {
  assert.equal(isEvalTrialMetadata({ "cayu:eval_trial": true }), true)
  assert.equal(isEvalTrialMetadata({ "cayu:eval_trial": "true" }), false)
  assert.equal(isEvalTrialMetadata({}), false)
  assert.equal(isEvalTrialMetadata(undefined), false)
})

test("shouldFollowRowClick leaves modified clicks and clicks on controls to the browser", () => {
  const plain = {
    button: 0,
    metaKey: false,
    ctrlKey: false,
    shiftKey: false,
    altKey: false,
    defaultPrevented: false,
    onControl: false,
  }
  assert.equal(shouldFollowRowClick(plain), true)
  assert.equal(shouldFollowRowClick({ ...plain, metaKey: true }), false)
  assert.equal(shouldFollowRowClick({ ...plain, ctrlKey: true }), false)
  assert.equal(shouldFollowRowClick({ ...plain, shiftKey: true }), false)
  assert.equal(shouldFollowRowClick({ ...plain, button: 1 }), false)
  assert.equal(shouldFollowRowClick({ ...plain, onControl: true }), false)
  assert.equal(shouldFollowRowClick({ ...plain, defaultPrevented: true }), false)
})
