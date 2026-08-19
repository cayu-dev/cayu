import assert from "node:assert/strict"
import test from "node:test"

import { providerOperationNeedsResolution } from "../src/lib/provider-operations.ts"

test("provider-operation controls disappear after a fail resolution", () => {
  assert.equal(
    providerOperationNeedsResolution({
      status: "provider_operation_unavailable",
      resolution_action: "fail",
      allowed_resolutions: [],
    }),
    false,
  )
})

test("provider-operation controls require unresolved positive authority", () => {
  assert.equal(
    providerOperationNeedsResolution({
      status: "provider_operation_unavailable",
      resolution_action: null,
      allowed_resolutions: ["fallback_retry", "fail"],
    }),
    true,
  )
  assert.equal(
    providerOperationNeedsResolution({
      status: "ambiguous_submission",
      resolution_action: null,
      allowed_resolutions: [],
    }),
    false,
  )
})
