import assert from "node:assert/strict"
import test from "node:test"
import {
  READING_END_SLACK,
  READING_POSITION_PREFIX,
  readingPositionFromScroll,
  readReadingPosition,
  writeReadingPosition,
} from "../src/lib/reading-position.ts"

function memoryStorage() {
  const values = new Map()
  return {
    getItem: (key) => (values.has(key) ? values.get(key) : null),
    setItem: (key, value) => values.set(key, String(value)),
    removeItem: (key) => values.delete(key),
    values,
  }
}

test("a scroll position records whether it follows the end", () => {
  assert.deepEqual(
    readingPositionFromScroll({ scrollTop: 120.6, scrollHeight: 2000, clientHeight: 600 }, "v"),
    { view: "v", top: 121, atEnd: false },
  )
  assert.equal(
    readingPositionFromScroll(
      { scrollTop: 1400 - READING_END_SLACK, scrollHeight: 2000, clientHeight: 600 },
      "v",
    ).atEnd,
    true,
  )
  assert.equal(
    readingPositionFromScroll({ scrollTop: 0, scrollHeight: 400, clientHeight: 600 }, "v").atEnd,
    true,
  )
})

test("a position is measured from its anchor, so content above it can change", () => {
  const metrics = { scrollTop: 640, scrollHeight: 3000, clientHeight: 600 }
  assert.equal(readingPositionFromScroll(metrics, "v", 180).top, 460)
  assert.equal(readingPositionFromScroll({ ...metrics, scrollTop: 100 }, "v", 180).top, -80)
  const storage = memoryStorage()
  writeReadingPosition("k", { view: "v", top: -80, atEnd: false }, storage)
  assert.equal(readReadingPosition("k", storage)?.top, -80)
})

test("positions round-trip per key and can be forgotten", () => {
  const storage = memoryStorage()
  assert.equal(readReadingPosition("session:a", storage), null)
  writeReadingPosition("session:a", { view: "a:all", top: 340, atEnd: false }, storage)
  assert.deepEqual(readReadingPosition("session:a", storage), {
    view: "a:all",
    top: 340,
    atEnd: false,
  })
  assert.equal(readReadingPosition("session:b", storage), null)
  writeReadingPosition("session:a", null, storage)
  assert.equal(storage.values.size, 0)
})

test("malformed, missing or blocked storage restores nothing", () => {
  const storage = memoryStorage()
  for (const raw of [
    "not json",
    "null",
    "[]",
    JSON.stringify({ view: "v", top: null, atEnd: false }),
    JSON.stringify({ view: "v", top: "12", atEnd: false }),
    JSON.stringify({ view: 3, top: 12, atEnd: false }),
    JSON.stringify({ view: "v", top: 12 }),
  ]) {
    storage.values.set(`${READING_POSITION_PREFIX}k`, raw)
    assert.equal(readReadingPosition("k", storage), null, raw)
  }
  assert.equal(readReadingPosition("k", null), null)
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
  assert.equal(readReadingPosition("k", blocked), null)
  assert.doesNotThrow(() => writeReadingPosition("k", { view: "v", top: 1, atEnd: false }, blocked))
})
