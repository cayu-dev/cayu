import { formatCurrencyWithCode } from "@/lib/format"
import type { UsageRollupResponse, UsageSessionCostSummary } from "@/lib/generated/server-api"

export const partialCost = (cost: UsageSessionCostSummary) =>
  cost.accuracy.kind !== "exact" ||
  cost.unpriced_model_steps !== "0" ||
  cost.unevaluated_model_steps !== "0" ||
  (cost.unpriced_auxiliary_attempts ?? "0") !== "0" ||
  cost.unpriced_reasons.length > 0

/** Present server estimates; never calculate prices or combine currencies here. */
export function usageCostText(cost: UsageSessionCostSummary | null | undefined): string {
  if (!cost) return "Unavailable"
  const partial = partialCost(cost)
  const amounts = cost.currencies
    .map((value) => formatCurrencyWithCode(value.total_cost, value.currency))
    .join(" · ")
  return amounts
    ? `${amounts}${partial ? " (partial)" : ""}`
    : partial
      ? "Unavailable"
      : "0 (no priced usage)"
}

export function validateCostRanking(data: UsageRollupResponse): UsageRollupResponse {
  const ranking = data.session_ranking
  if (
    ranking?.requested_order !== "cost" ||
    !["tokens", "cost"].includes(ranking.effective_order) ||
    ranking.effective_order !== data.session_breakdown?.order ||
    (ranking.effective_order === "tokens"
      ? !ranking.unavailable_reason
      : !!ranking.unavailable_reason)
  )
    throw new Error("The server did not return a valid session cost ranking.")
  return data
}

export function rankingExplanation(data: UsageRollupResponse, supported: boolean): string | null {
  if (!supported)
    return "Cost ranking is not supported by this agent. Sessions are ranked by tokens."
  switch (data.session_ranking?.unavailable_reason) {
    case "pricing_not_configured":
      return "Sessions are ranked by tokens because no price book is configured."
    case "prices_incomplete":
      return "Sessions are ranked by tokens because some usage has no complete price."
    case "mixed_currencies":
      return "Sessions are ranked by tokens because costs use different currencies."
    case "candidate_limit":
      return "Sessions are ranked by tokens because this range has too many sessions to compare all costs. Narrow the filters or range."
    case "pricing_projection_incomplete":
    case "candidate_byte_limit":
      return "Sessions are ranked by tokens because complete cost data is unavailable for this range. Narrow the filters or range."
    case null:
    case undefined:
      return null
    default:
      return "Cost ranking is unavailable for this range. Sessions are ranked by tokens."
  }
}
