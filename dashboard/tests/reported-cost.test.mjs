import assert from "node:assert/strict"
import test from "node:test"
import { reportedCostText } from "../src/lib/reported-cost.ts"

test("reported amounts stay exact and distinct from pending and estimates", () => {
  assert.equal(
    reportedCostText({ status: "reported", currency: "USD", cost: "0.000000001" }),
    "0.000000001 USD",
  )
  assert.equal(
    reportedCostText({ status: "reported", currency: "USD", cost: "0.000000000" }),
    "0.000000000 USD",
  )
  assert.match(
    reportedCostText({ status: "pending", currency: "USD", cost: null }),
    /Pending.*unknown/,
  )
  assert.match(
    reportedCostText({ status: "unavailable", currency: "USD", cost: null }),
    /Unavailable.*unknown/,
  )
})
