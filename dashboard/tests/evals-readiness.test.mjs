import assert from "node:assert/strict"
import test from "node:test"

import {
  EVALS_READINESS_OPERATIONS,
  evalsReadinessReasonText,
  evalsReadinessStateLabel,
} from "../src/lib/evals-readiness.ts"

test("Evals readiness exposes every operation in a stable product order", () => {
  assert.deepEqual(
    EVALS_READINESS_OPERATIONS.map(([operation]) => operation),
    [
      "captured_evaluation",
      "catalog_read",
      "catalog_write",
      "captured_result_persistence",
      "scenario_conversion",
      "fresh_launch",
      "cancellation",
      "comparison",
      "reports",
    ],
  )
  assert.equal(
    EVALS_READINESS_OPERATIONS.find(([operation]) => operation === "fresh_launch")?.[1],
    "Current-app trials",
  )
})

test("Evals readiness copy distinguishes ready, deployment-gated, and unavailable operations", () => {
  assert.equal(evalsReadinessStateLabel({ state: "ready", reason_code: null }), "Ready")
  assert.equal(
    evalsReadinessReasonText({ state: "ready", reason_code: null }),
    "Available in this deployment; the API still enforces authorization and request preconditions.",
  )
  assert.equal(
    evalsReadinessStateLabel({
      state: "gated",
      reason_code: "eval_store_not_configured",
    }),
    "Not ready",
  )
  assert.equal(
    evalsReadinessReasonText({
      state: "gated",
      reason_code: "eval_store_not_configured",
    }),
    "For mount_cayu, pass access=AuthenticatedAccess(...) and complete durable Evals wiring through evals=EvalsConfig(target=target, store=eval_store), where eval_store is a SQLiteEvalStore or PostgresEvalStore.",
  )
  assert.equal(
    evalsReadinessReasonText({
      state: "gated",
      reason_code: "eval_target_not_configured",
    }),
    "For mount_cayu, pass access=AuthenticatedAccess(...) and complete durable Evals wiring through evals=EvalsConfig(target=target, store=eval_store), where target is a CorpusTarget for the exact mounted CayuApp.",
  )
  assert.equal(
    evalsReadinessStateLabel({
      state: "unsupported",
      reason_code: "captured_result_persistence_not_available",
    }),
    "Unavailable",
  )
  assert.equal(
    evalsReadinessReasonText({
      state: "unsupported",
      reason_code: "session_lineage_not_supported",
    }),
    "The session store cannot provide the session lineage required for captured evaluation.",
  )
  assert.equal(
    evalsReadinessStateLabel({
      state: "unsupported",
      reason_code: "scenario_v2_not_available",
    }),
    "Unavailable",
  )
  assert.equal(
    evalsReadinessReasonText({
      state: "unsupported",
      reason_code: "scenario_v2_not_available",
    }),
    "Scenario-v2 conversion is unavailable in this deployment.",
  )
})
