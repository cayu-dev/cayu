import type { ReportedCostObservation } from "./generated/server-api"

export function reportedCostText(row: ReportedCostObservation): string {
  if (row.status === "reported" && row.currency === "USD" && row.cost !== null) {
    // Do not round a small positive charge into a displayed zero.
    return `${row.cost} USD`
  }
  return row.status === "pending" ? "Pending · amount unknown" : "Unavailable · amount unknown"
}
