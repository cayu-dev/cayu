import assert from "node:assert/strict"
import test from "node:test"
import {
  clearSessionDraft,
  readSessionDraft,
  SESSION_DRAFT_PREFIX,
  writeSessionDraft,
} from "../src/lib/session-draft.ts"

function memoryStorage() {
  const values = new Map()
  return {
    getItem: (key) => (values.has(key) ? values.get(key) : null),
    setItem: (key, value) => values.set(key, String(value)),
    removeItem: (key) => values.delete(key),
    values,
  }
}

test("drafts are kept per session and cleared when sent", () => {
  const storage = memoryStorage()
  assert.equal(readSessionDraft("session-a", storage), "")
  writeSessionDraft("session-a", "  half a thought\n", storage)
  writeSessionDraft("session-b", "another", storage)
  assert.equal(storage.values.get(`${SESSION_DRAFT_PREFIX}session-a`), "  half a thought\n")
  assert.equal(readSessionDraft("session-a", storage), "  half a thought\n")
  assert.equal(readSessionDraft("session-b", storage), "another")
  clearSessionDraft("session-a", storage)
  assert.equal(readSessionDraft("session-a", storage), "")
  assert.equal(storage.values.has(`${SESSION_DRAFT_PREFIX}session-a`), false)
  writeSessionDraft("session-b", "", storage)
  assert.equal(storage.values.size, 0)
})

test("missing or blocked storage keeps drafts in the page only", () => {
  assert.equal(readSessionDraft("session-a", null), "")
  assert.doesNotThrow(() => writeSessionDraft("session-a", "text", null))
  const blocked = {
    getItem() {
      throw new Error("blocked")
    },
    setItem() {
      throw new Error("quota")
    },
    removeItem() {
      throw new Error("blocked")
    },
  }
  assert.equal(readSessionDraft("session-a", blocked), "")
  assert.doesNotThrow(() => writeSessionDraft("session-a", "text", blocked))
  assert.doesNotThrow(() => clearSessionDraft("session-a", blocked))
})
