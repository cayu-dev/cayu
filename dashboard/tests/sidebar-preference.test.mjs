import assert from "node:assert/strict"
import test from "node:test"
import {
  readSidebarHidden,
  SIDEBAR_KEY,
  writeSidebarHidden,
} from "../src/lib/sidebar-preference.ts"

function memoryStorage() {
  const values = new Map()
  return {
    getItem: (key) => (values.has(key) ? values.get(key) : null),
    setItem: (key, value) => values.set(key, String(value)),
    values,
  }
}

test("the sidebar preference round-trips through storage", () => {
  const storage = memoryStorage()
  assert.equal(readSidebarHidden(storage), false)
  writeSidebarHidden(true, storage)
  assert.equal(storage.values.get(SIDEBAR_KEY), "hidden")
  assert.equal(readSidebarHidden(storage), true)
  writeSidebarHidden(false, storage)
  assert.equal(readSidebarHidden(storage), false)
})

test("missing, unknown or blocked storage shows the sidebar", () => {
  assert.equal(readSidebarHidden(null), false)
  const unknown = memoryStorage()
  unknown.setItem(SIDEBAR_KEY, "sideways")
  assert.equal(readSidebarHidden(unknown), false)
  const blocked = {
    getItem() {
      throw new Error("blocked")
    },
    setItem() {
      throw new Error("blocked")
    },
  }
  assert.equal(readSidebarHidden(blocked), false)
  assert.doesNotThrow(() => writeSidebarHidden(true, blocked))
})
