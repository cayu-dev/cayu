import assert from "node:assert/strict"
import test from "node:test"

import {
  formatCount,
  formatCurrency,
  formatCurrencyWithCode,
  formatDecimal,
  sumCounts,
} from "../src/lib/format.ts"

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

test("currency formatting keeps small estimates visible and the currency unambiguous", () => {
  assert.equal(formatCurrency("0.00457314", "USD", "en-US"), "$0.0046")
  assert.equal(formatCurrency("0.00001", "USD", "en-US"), "<$0.0001")
  assert.equal(formatCurrency("0.009", "USD", "en-US"), "$0.01")
  assert.equal(formatCurrency("0.00457314", "KWD", "en-US"), "KWD\u00a00.005")
  assert.equal(formatCurrency("1.2", "USD", "en-US"), "$1.20")
  assert.equal(formatCurrency("1234.5", "EUR", "de-DE"), "1.234,50\u00a0€")
  assert.equal(formatCurrency("0.00001", "NOT_A_CURRENCY", "en-US"), "0.00001 NOT_A_CURRENCY")
})

test("currency formatting preserves exact bounded exponent notation", () => {
  assert.equal(formatCurrency("1E-7", "USD", "en-US"), "<$0.0001")
  assert.equal(formatCurrency("4.57314E-3", "USD", "en-US"), "$0.0046")
  assert.equal(formatCurrency("12E+2", "USD", "en-US"), "$1,200.00")
  assert.equal(formatCurrency("1E+0002", "USD", "en-US"), "$100.00")
  assert.equal(formatDecimal("1E-7", "en-US"), "0.0000001")
  assert.equal(formatDecimal("12E+2", "en-US"), "1,200")
  assert.equal(formatCurrency("1E+4097", "USD", "en-US"), "1E+4097 USD")
  assert.equal(formatCurrency("1E-4097", "USD", "en-US"), "1E-4097 USD")
  assert.equal(
    formatCurrency(`1E+${"9".repeat(4_097)}`, "USD", "en-US"),
    `1E+${"9".repeat(4_097)} USD`,
  )

  const usageResponse = {
    cost: { currencies: [{ currency: "USD", model_steps: "1", total_cost: "1E-7" }] },
  }
  const currency = usageResponse.cost.currencies[0]
  assert.equal(formatCurrency(currency.total_cost, currency.currency, "en-US"), "<$0.0001")
})

test("code formatting preserves colliding multi-currency identities", () => {
  assert.equal(formatCurrencyWithCode("1", "USD", "en-US"), "USD\u00a01.00")
  assert.equal(formatCurrencyWithCode("2", "CAD", "en-US"), "CAD\u00a02.00")
  assert.equal(formatCurrencyWithCode("1E-7", "USD", "en-US"), "<USD\u00a00.0001")
  assert.equal(formatCurrencyWithCode("1E-7", "CAD", "en-US"), "<CAD\u00a00.0001")
})
