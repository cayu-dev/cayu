import assert from "node:assert/strict"
import test from "node:test"
import { InboxSettleError, inboxApiFromCayuClient } from "../src/parts/inbox/cayu-client.ts"
import { isProfileChangedConflict } from "../src/parts/inbox/inbox-adoption.ts"

const fakeClient = (settle) => {
  const calls = []
  return {
    calls,
    listInbox: async (options) => {
      calls.push(["listInbox", options])
      return { items: [], issues: [], next_cursor: null, has_more: false }
    },
    getInboxCounts: async (options) => {
      calls.push(["getInboxCounts", options])
      return { counts: null, total_count: null, complete: false, unavailable_reason: null }
    },
    settleInboxItem: async (id, body) => {
      calls.push(["settleInboxItem", id, body])
      return settle
    },
  }
}

test("reads pass filters, cursor and signal through to client.js", async () => {
  const client = fakeClient()
  const api = inboxApiFromCayuClient(client)
  const signal = new AbortController().signal
  await api.list({ q: "pay", kind: "approval", limit: 25, cursor: null }, signal)
  await api.list({ cursor: "c2" })
  await api.counts({ session_id: "s1" }, signal)
  assert.deepEqual(client.calls, [
    ["listInbox", { q: "pay", kind: "approval", limit: 25, cursor: undefined, signal }],
    ["listInbox", { cursor: "c2", signal: undefined }],
    ["getInboxCounts", { session_id: "s1", signal }],
  ])
  assert.equal(api.watch, undefined)
})

test("a settled item resolves with its state", async () => {
  const item = { id: "i1", kind: "approval", status: "settled" }
  const client = fakeClient({ settled: true, replayed: false, item })
  const body = { decision: "approve", reason: null }
  assert.deepEqual(await inboxApiFromCayuClient(client).settle("i1", body), item)
  assert.deepEqual(client.calls, [["settleInboxItem", "i1", body]])
})

test("settled: false rejects with the status and code, which adoption reads", async () => {
  const item = { id: "i1", kind: "approval", status: "pending" }
  const api = inboxApiFromCayuClient(
    fakeClient({ settled: false, code: "profile_changed", message: "Updated.", item }),
  )
  const error = await api.settle("i1", { decision: "approve" }).then(
    () => assert.fail("settle resolved"),
    (rejection) => rejection,
  )
  assert.ok(error instanceof InboxSettleError)
  assert.equal(error.status, 409)
  assert.equal(error.code, "profile_changed")
  assert.equal(error.message, "Updated.")
  assert.deepEqual(error.item, item)
  assert.equal(isProfileChangedConflict(error), true)
})

test("watch follows client.js's watchInbox and returns its stop", () => {
  const seen = []
  const client = {
    ...fakeClient(),
    watchInbox(options) {
      seen.push(this === client, options.q)
      options.onChange({ items: [], issues: [], complete: true })
      return () => seen.push("stopped")
    },
  }
  let changes = 0
  const stop = inboxApiFromCayuClient(client).watch({ q: "x" }, () => {
    changes += 1
  })
  stop()
  assert.deepEqual(seen, [true, "x", "stopped"])
  assert.equal(changes, 1)
})
