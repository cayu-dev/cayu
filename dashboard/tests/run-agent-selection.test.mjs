import assert from "node:assert/strict"
import test from "node:test"

import { reconcileRunAgentSelection } from "../src/lib/run-agent-selection.ts"

test("a single registered agent is selected automatically", () => {
  assert.equal(reconcileRunAgentSelection("", [{ name: "reviewer" }]), "reviewer")
})

test("multiple registered agents require an explicit selection", () => {
  const agents = [{ name: "assistant" }, { name: "reviewer" }]

  assert.equal(reconcileRunAgentSelection("", agents), "")
  assert.equal(reconcileRunAgentSelection("reviewer", agents), "reviewer")
})

test("a selection that is no longer registered fails closed", () => {
  assert.equal(
    reconcileRunAgentSelection("removed", [{ name: "assistant" }, { name: "reviewer" }]),
    "",
  )
  assert.equal(reconcileRunAgentSelection("removed", [{ name: "reviewer" }]), "reviewer")
  assert.equal(reconcileRunAgentSelection("removed", []), "")
})

test("JSON-like agent names remain exact opaque identities", () => {
  for (const name of ["123", "true", "null"]) {
    assert.equal(reconcileRunAgentSelection(name, [{ name }, { name: "other" }]), name)
  }
})
