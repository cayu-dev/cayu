import assert from "node:assert/strict"
import test from "node:test"

import { evalSuiteTrialPolicy } from "../src/lib/eval-trial-policy.ts"

test("a trial policy is the server's own, revision included", async () => {
  // `EvalSuiteTrialPolicyV1.create(trial_count=3, max_concurrency=2).model_dump(mode="json")`
  assert.deepEqual(await evalSuiteTrialPolicy(3, 2), {
    schema_version: 1,
    revision: "sha256:7caa7d8ce25323c0ac88629861529ceda21a9fd17c998868092a14e18474f7e9",
    trial_count: 3,
    minimum_passed_trials: 3,
    max_concurrency: 2,
    require_zero_runtime_errors: true,
    require_zero_evaluator_errors: true,
    require_complete_required_evidence: true,
  })
  // The policy the server assumes for a request without one.
  assert.equal(
    (await evalSuiteTrialPolicy(1, 1)).revision,
    "sha256:6eec71d36aca93895f962821d4e52c506d24228ad8bc162dd3f7b8a3a5a271db",
  )
})
