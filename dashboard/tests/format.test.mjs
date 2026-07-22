import assert from "node:assert/strict"
import test from "node:test"

import { formatCount, formatDecimal, sumCounts } from "../src/lib/format.ts"

test("aggregate formatting preserves integers beyond JavaScript's safe range", () => {
  assert.equal(formatCount("9007199254740993"), 9007199254740993n.toLocaleString())
  assert.equal(sumCounts("9007199254740993", "9007199254740993", "1"), "18014398509481987")
})

test("exact decimal strings are displayed without floating-point rounding", () => {
  assert.equal(formatDecimal("9007199254740993", "en-US"), "9,007,199,254,740,993")
  assert.equal(formatDecimal("0.0000001", "en-US"), "0.0000001")
  assert.equal(formatDecimal("1234.500000", "en-US"), "1,234.5")
  assert.equal(formatDecimal("1234.500000", "de-DE"), "1.234,5")
  assert.equal(formatDecimal("-0.500000", "de-DE"), "-0,5")
  assert.equal(formatDecimal("1234.500000", "ar-EG"), "١٬٢٣٤٫٥")
})
