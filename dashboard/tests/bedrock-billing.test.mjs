import assert from "node:assert/strict"
import test from "node:test"

import { bedrockBillingRows } from "../src/lib/bedrock-billing.ts"

function lineItem(identity, { priced = true, cost = "1.25", reason = null } = {}) {
  return {
    billing_identity: identity,
    pricing_model: priced ? identity.resource_id : null,
    priced,
    total_cost: cost,
    missing_pricing_reason: reason,
  }
}

test("Bedrock billing rows preserve exact scope, region, and tier buckets", () => {
  const globalDefault = {
    provider_name: "bedrock",
    resource_id: "global.anthropic.claude-sonnet-4-6",
    request_evidence: {
      source_region: "us-east-1",
      resource_type: "inference_profile",
      profile_scope: "global",
      requested_service_tier: "default",
    },
    completion_evidence: { effective_service_tier: "default" },
  }
  const rows = bedrockBillingRows([
    lineItem(globalDefault, { cost: "1.25" }),
    lineItem(globalDefault, { cost: "2.75" }),
    lineItem({
      ...globalDefault,
      request_evidence: { ...globalDefault.request_evidence, source_region: "us-west-2" },
    }),
    lineItem({
      ...globalDefault,
      request_evidence: { ...globalDefault.request_evidence, requested_service_tier: "flex" },
    }),
    lineItem(
      {
        ...globalDefault,
        resource_id: "us.anthropic.claude-sonnet-4-6",
        request_evidence: {
          ...globalDefault.request_evidence,
          profile_scope: "geographic",
        },
      },
      { priced: false, cost: "0", reason: "no matching model pricing" },
    ),
  ])

  assert.equal(rows.length, 4)
  const aggregate = rows.find(
    (row) =>
      row.identity.invoked_model === globalDefault.resource_id &&
      row.identity.source_region === globalDefault.request_evidence.source_region &&
      row.identity.requested_service_tier === "default",
  )
  assert.deepEqual(
    {
      priced: aggregate?.priced,
      unpriced: aggregate?.unpriced,
      totalCost: aggregate?.totalCost,
    },
    { priced: 2, unpriced: 0, totalCost: "4" },
  )
  const unpriced = rows.find((row) => row.identity.profile_scope === "geographic")
  assert.equal(unpriced?.missingReason, "no matching model pricing")
  assert.equal(unpriced?.unpriced, 1)
})
