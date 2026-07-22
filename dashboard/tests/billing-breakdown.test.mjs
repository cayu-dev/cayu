import assert from "node:assert/strict"
import test from "node:test"

import { billingCostRows, billingIdentityCoverage } from "../src/lib/billing-breakdown.ts"

function billingGroup(identity, { priced = true, cost = "1.25", reason = null } = {}) {
  return {
    billing_identity: identity,
    pricing_provider_name: priced ? identity.provider_name : null,
    pricing_model: priced ? identity.resource_id : null,
    priced,
    model_steps: "1",
    currency: priced ? "USD" : null,
    total_cost: cost,
    missing_pricing_reason: reason,
  }
}

test("billing rows preserve exact Bedrock scope, region, and tier buckets", () => {
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
  const rows = billingCostRows([
    billingGroup(globalDefault, { cost: "4" }),
    billingGroup({
      ...globalDefault,
      request_evidence: { ...globalDefault.request_evidence, source_region: "us-west-2" },
    }),
    billingGroup({
      ...globalDefault,
      request_evidence: { ...globalDefault.request_evidence, requested_service_tier: "flex" },
    }),
    billingGroup(
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
  const priced = rows.find(
    (row) =>
      row.identity.invoked_model === globalDefault.resource_id &&
      row.identity.source_region === globalDefault.request_evidence.source_region &&
      row.identity.requested_service_tier === "default",
  )
  assert.deepEqual(
    {
      priced: priced?.priced,
      unpriced: priced?.unpriced,
      totalCost: priced?.totalCost,
      currency: priced?.currency,
      pricingProvider: priced?.pricingProvider,
    },
    {
      priced: "1",
      unpriced: "0",
      totalCost: "4",
      currency: "USD",
      pricingProvider: "bedrock",
    },
  )
  const unpriced = rows.find((row) => row.identity.profile_scope === "geographic")
  assert.equal(unpriced?.missingReason, "no matching model pricing")
  assert.equal(unpriced?.unpriced, "1")
  assert.equal(unpriced?.currency, null)
})

test("billing rows retain unresolved Bedrock identities", () => {
  const rows = billingCostRows([
    billingGroup(
      {
        provider_name: "bedrock",
        resource_id: "anthropic.claude-sonnet-4-6-v1:0",
        request_evidence: {},
        completion_evidence: {},
      },
      { priced: false, cost: "0", reason: "no matching model pricing" },
    ),
  ])

  assert.equal(rows.length, 1)
  assert.equal(rows[0]?.identity.resource_type, null)
  assert.equal(rows[0]?.identity.requested_service_tier, null)
  assert.equal(rows[0]?.unpriced, "1")
})

test("billing rows do not discard retained identities from other providers", () => {
  const rows = billingCostRows([
    billingGroup({
      provider_name: "openai",
      resource_id: "gpt-5.5",
      request_evidence: {},
      completion_evidence: {},
    }),
    billingGroup({
      provider_name: "bedrock",
      resource_id: "global.anthropic.claude-sonnet-4-6",
      request_evidence: { source_region: "us-east-1" },
      completion_evidence: {},
    }),
  ])

  assert.deepEqual(
    rows.map((row) => row.identity.provider_name),
    ["openai", "bedrock"],
  )
  assert.equal(rows[0]?.identity.source_region, null)
  assert.equal(rows[1]?.identity.source_region, "us-east-1")
})

test("billing identity coverage remains exact beyond JavaScript's safe integer range", () => {
  assert.deepEqual(
    billingIdentityCoverage({
      evaluated_model_steps: "9007199254740993",
      billing_breakdown: { identified_model_steps: "2" },
    }),
    {
      evaluated: "9007199254740993",
      identified: "2",
      missing: "9007199254740991",
    },
  )
})

test("billing identity coverage distinguishes complete and absent identity reporting", () => {
  assert.deepEqual(
    billingIdentityCoverage({
      evaluated_model_steps: "2",
      billing_breakdown: { identified_model_steps: "2" },
    }),
    { evaluated: "2", identified: "2", missing: "0" },
  )
  assert.deepEqual(
    billingIdentityCoverage({
      evaluated_model_steps: "3",
      billing_breakdown: { identified_model_steps: "0" },
    }),
    { evaluated: "3", identified: "0", missing: "3" },
  )
})

test("billing identity coverage rejects inconsistent response counters", () => {
  assert.equal(
    billingIdentityCoverage({
      evaluated_model_steps: "1",
      billing_breakdown: { identified_model_steps: "2" },
    }),
    null,
  )
  assert.equal(
    billingIdentityCoverage({
      evaluated_model_steps: "not-an-integer",
      billing_breakdown: { identified_model_steps: "0" },
    }),
    null,
  )
})
