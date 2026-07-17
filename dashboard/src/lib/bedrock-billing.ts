import { addDecimalStrings } from "./decimal.ts"
import type { BillingIdentity, CostLineItem } from "./generated/server-api"

export type BedrockBillingView = {
  provider_name: "bedrock"
  invoked_model: string
  source_region: string | null
  resource_type: string
  profile_scope: string | null
  requested_service_tier: string
  effective_service_tier: string | null
}

export type BedrockBillingRow = {
  identity: BedrockBillingView
  pricingModel: string | null
  priced: number
  unpriced: number
  totalCost: string
  missingReason: string | null
}

export function bedrockBillingRows(lineItems: CostLineItem[]): BedrockBillingRow[] {
  const buckets = new Map<string, BedrockBillingRow>()
  for (const item of lineItems) {
    const identity = bedrockBillingIdentity(item.billing_identity)
    if (!identity) continue
    const key = JSON.stringify([
      identity.invoked_model,
      identity.source_region,
      identity.resource_type,
      identity.profile_scope,
      identity.requested_service_tier,
      identity.effective_service_tier,
    ])
    const existing = buckets.get(key)
    buckets.set(key, {
      identity,
      pricingModel: item.pricing_model ?? existing?.pricingModel ?? null,
      priced: (existing?.priced ?? 0) + (item.priced ? 1 : 0),
      unpriced: (existing?.unpriced ?? 0) + (item.priced ? 0 : 1),
      totalCost: addDecimalStrings(existing?.totalCost ?? "0", item.total_cost),
      missingReason: item.missing_pricing_reason ?? existing?.missingReason ?? null,
    })
  }
  return [...buckets.values()].sort((left, right) =>
    JSON.stringify(left.identity).localeCompare(JSON.stringify(right.identity)),
  )
}

function bedrockBillingIdentity(
  identity: BillingIdentity | null | undefined,
): BedrockBillingView | null {
  if (identity?.provider_name !== "bedrock") return null
  const request = identity.request_evidence ?? {}
  const completion = identity.completion_evidence ?? {}
  const resourceType = stringValue(request.resource_type)
  const requestedTier = stringValue(request.requested_service_tier)
  if (!resourceType || !requestedTier) return null
  return {
    provider_name: "bedrock",
    invoked_model: identity.resource_id,
    source_region: stringValue(request.source_region),
    resource_type: resourceType,
    profile_scope: stringValue(request.profile_scope),
    requested_service_tier: requestedTier,
    effective_service_tier: stringValue(completion.effective_service_tier),
  }
}

function stringValue(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null
}
