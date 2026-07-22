import type { UsageBillingCostGroup, UsageCostRollup } from "./generated/server-api"

export type BillingIdentityView = {
  provider_name: string
  invoked_model: string
  source_region: string | null
  resource_type: string | null
  profile_scope: string | null
  requested_service_tier: string | null
  effective_service_tier: string | null
}

export type BillingCostRow = {
  identity: BillingIdentityView
  pricingProvider: string | null
  pricingModel: string | null
  priced: string
  unpriced: string
  currency: string | null
  totalCost: string
  missingReason: string | null
}

export type BillingIdentityCoverage = {
  evaluated: string
  identified: string
  missing: string
}

export function billingIdentityCoverage(cost: UsageCostRollup): BillingIdentityCoverage | null {
  const evaluated = nonnegativeInteger(cost.evaluated_model_steps)
  const identified = nonnegativeInteger(cost.billing_breakdown.identified_model_steps)
  if (evaluated === null || identified === null || identified > evaluated) return null
  return {
    evaluated: evaluated.toString(),
    identified: identified.toString(),
    missing: (evaluated - identified).toString(),
  }
}

export function billingCostRows(groups: UsageBillingCostGroup[]): BillingCostRow[] {
  return groups.map((group) => ({
    identity: billingIdentity(group),
    pricingProvider: group.pricing_provider_name ?? null,
    pricingModel: group.pricing_model ?? null,
    priced: group.priced ? group.model_steps : "0",
    unpriced: group.priced ? "0" : group.model_steps,
    currency: group.currency ?? null,
    totalCost: group.total_cost,
    missingReason: group.missing_pricing_reason ?? null,
  }))
}

function billingIdentity(group: UsageBillingCostGroup): BillingIdentityView {
  const identity = group.billing_identity
  const isBedrock = identity.provider_name === "bedrock"
  const request = isBedrock ? (identity.request_evidence ?? {}) : {}
  const completion = isBedrock ? (identity.completion_evidence ?? {}) : {}
  return {
    provider_name: identity.provider_name,
    invoked_model: identity.resource_id,
    source_region: stringValue(request.source_region),
    resource_type: stringValue(request.resource_type),
    profile_scope: stringValue(request.profile_scope),
    requested_service_tier: stringValue(request.requested_service_tier),
    effective_service_tier: stringValue(completion.effective_service_tier),
  }
}

function stringValue(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null
}

function nonnegativeInteger(value: string): bigint | null {
  if (!/^\d+$/.test(value)) return null
  try {
    return BigInt(value)
  } catch {
    return null
  }
}
