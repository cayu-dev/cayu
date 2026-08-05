import assert from "node:assert/strict"
import test from "node:test"

import { billingCostRows, billingIdentityBreakdownState } from "../src/lib/billing-breakdown.ts"

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

test("billing identity breakdown is not applicable to priced steps without identities", () => {
  assert.deepEqual(
    billingIdentityBreakdownState({
      evaluated_model_steps: "1",
      priced_model_steps: "1",
      unpriced_model_steps: "0",
      unevaluated_model_steps: "0",
      billing_breakdown: {
        identified_model_steps: "0",
        groups: [],
        remainder: null,
        accuracy: { kind: "exact", limit: null, reason: null },
      },
    }),
    { kind: "not-applicable" },
  )
})

test("mixed billing identity detail reports only identity-bearing steps", () => {
  assert.deepEqual(
    billingIdentityBreakdownState({
      evaluated_model_steps: "5",
      billing_breakdown: {
        identified_model_steps: "1",
        groups: [
          billingGroup({
            provider_name: "bedrock",
            resource_id: "global.anthropic.claude-sonnet-4-6",
            request_evidence: { source_region: "us-east-1" },
            completion_evidence: {},
          }),
        ],
        remainder: null,
        accuracy: { kind: "exact", limit: null, reason: null },
      },
    }),
    {
      kind: "available",
      evaluated: "5",
      identityBearing: "1",
      hasIdentityDetail: true,
    },
  )
})

test("truncated billing identity detail remains available for its accuracy notice", () => {
  assert.deepEqual(
    billingIdentityBreakdownState({
      evaluated_model_steps: "3",
      billing_breakdown: {
        identified_model_steps: "3",
        groups: [
          billingGroup({
            provider_name: "bedrock",
            resource_id: "global.anthropic.claude-sonnet-4-6",
            request_evidence: { source_region: "us-east-1" },
            completion_evidence: {},
          }),
        ],
        remainder: {
          group_count: "2",
          model_steps: "2",
          priced_model_steps: "2",
          unpriced_model_steps: "0",
        },
        accuracy: {
          kind: "truncated",
          limit: 1,
          reason: "Billing identity groups exceed group_limit.",
        },
      },
    }),
    {
      kind: "available",
      evaluated: "3",
      identityBearing: "3",
      hasIdentityDetail: true,
    },
  )
})

test("truncated zero-identity breakdown remains available for its accuracy notice", () => {
  assert.deepEqual(
    billingIdentityBreakdownState({
      evaluated_model_steps: "1",
      billing_breakdown: {
        identified_model_steps: "0",
        groups: [],
        remainder: null,
        accuracy: {
          kind: "truncated",
          limit: 20,
          reason: "Billing identity input detail exceeded its bounded collection limit.",
        },
      },
    }),
    {
      kind: "available",
      evaluated: "1",
      identityBearing: "0",
      hasIdentityDetail: false,
    },
  )
})

test("billing identity state remains exact beyond JavaScript's safe integer range", () => {
  assert.deepEqual(
    billingIdentityBreakdownState({
      evaluated_model_steps: "9007199254740993",
      billing_breakdown: {
        identified_model_steps: "2",
        groups: [],
        remainder: null,
        accuracy: { kind: "exact", limit: null, reason: null },
      },
    }),
    {
      kind: "available",
      evaluated: "9007199254740993",
      identityBearing: "2",
      hasIdentityDetail: false,
    },
  )
})

test("billing identity state exposes inconsistent response counters", () => {
  assert.deepEqual(
    billingIdentityBreakdownState({
      evaluated_model_steps: "1",
      billing_breakdown: {
        identified_model_steps: "2",
        groups: [],
        remainder: null,
        accuracy: { kind: "exact", limit: null, reason: null },
      },
    }),
    { kind: "inconsistent", hasIdentityDetail: false },
  )
  assert.deepEqual(
    billingIdentityBreakdownState({
      evaluated_model_steps: "not-an-integer",
      billing_breakdown: {
        identified_model_steps: "0",
        groups: [],
        remainder: null,
        accuracy: { kind: "exact", limit: null, reason: null },
      },
    }),
    { kind: "inconsistent", hasIdentityDetail: false },
  )
})
