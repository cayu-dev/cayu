import assert from "node:assert/strict"
import test from "node:test"
import { validateSessionDetailSearch } from "../src/lib/session-detail-search.ts"
import {
  readAsConversation,
  rememberConversation,
  SESSION_VIEW_PREFIX,
  sessionView,
} from "../src/lib/session-view.ts"

const open = (search, readAsConversation, waiting) =>
  sessionView(validateSessionDetailSearch(search), { readAsConversation, waiting })

test("a session opens on Trace unless the reader needs Conversation", () => {
  assert.equal(open({}, false, false), "trace")
  assert.equal(open({}, false, true), "conversation")
  assert.equal(open({}, false, undefined), undefined)
})

test("an explicit view or conversation filters in the link win", () => {
  assert.equal(open({ view: "trace" }, true, true), "trace")
  assert.equal(open({ view: "conversation" }, false, false), "conversation")
  assert.equal(open({ transcript_role: "user" }, false, false), "conversation")
  assert.equal(open({ include_thinking: "false" }, false, undefined), "conversation")
  assert.equal(open({ event_type: "model.completed" }, true, true), "trace")
})

test("returning to a session read as a conversation opens Conversation again", () => {
  assert.equal(open({}, true, false), "conversation")
  assert.equal(open({}, true, undefined), "conversation")
})

test("only Conversation is remembered, per session, and storage failures are tolerated", () => {
  const values = new Map()
  const storage = {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value),
    removeItem: (key) => values.delete(key),
  }
  assert.equal(readAsConversation("s1", storage), false)
  rememberConversation("s1", true, storage)
  assert.equal(values.get(`${SESSION_VIEW_PREFIX}s1`), "conversation")
  assert.equal(readAsConversation("s1", storage), true)
  assert.equal(readAsConversation("s2", storage), false)
  // Choosing Trace forgets it, so the session can open Conversation again once it waits.
  rememberConversation("s1", false, storage)
  assert.equal(readAsConversation("s1", storage), false)
  assert.equal(open({}, readAsConversation("s1", storage), true), "conversation")
  const broken = {
    getItem: () => {
      throw new Error("blocked")
    },
    setItem: () => {
      throw new Error("full")
    },
    removeItem: () => {
      throw new Error("blocked")
    },
  }
  assert.equal(readAsConversation("s1", broken), false)
  rememberConversation("s1", true, broken)
  assert.equal(readAsConversation("s1", null), false)
})
