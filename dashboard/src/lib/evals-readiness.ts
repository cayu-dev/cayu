import type { EvalsOperationReadiness, EvalsReadiness } from "./generated/server-api"

type EvalsReadinessReasonCode = NonNullable<EvalsOperationReadiness["reason_code"]>

export const EVALS_READINESS_OPERATIONS = [
  ["captured_evaluation", "Captured evaluation"],
  ["catalog_read", "Catalog"],
  ["catalog_write", "Import and save"],
  ["captured_result_persistence", "Captured results"],
  ["scenario_conversion", "Production scenarios"],
  ["fresh_launch", "Fresh trials"],
  ["cancellation", "Cancellation"],
  ["comparison", "Comparison"],
  ["reports", "Reports"],
] as const satisfies ReadonlyArray<readonly [keyof EvalsReadiness, string]>

const REASON_TEXT: Record<EvalsReadinessReasonCode, string> = {
  evaluation_promotion_not_configured:
    "Captured-session evaluation is not assembled in this deployment; automatic assembly is planned.",
  terminal_evidence_not_supported:
    "The session store cannot provide the terminal evidence required for captured evaluation.",
  session_lineage_not_supported:
    "The session store cannot provide the session lineage required for captured evaluation.",
  eval_store_not_configured:
    "Durable Evals storage is not assembled in this deployment; automatic assembly is planned.",
  eval_target_not_configured:
    "No approved fresh-execution target is assembled in this deployment; automatic discovery is planned.",
  captured_result_persistence_not_available:
    "Saving captured results is planned for a future Cayu release.",
  scenario_v2_not_available:
    "Multi-stage production scenarios are planned for a future Cayu release.",
}

export function evalsReadinessStateLabel(readiness: EvalsOperationReadiness): string {
  if (readiness.state === "ready") return "Ready"
  if (readiness.state === "gated") return "Not ready"
  if (
    readiness.reason_code === "captured_result_persistence_not_available" ||
    readiness.reason_code === "scenario_v2_not_available"
  ) {
    return "Planned"
  }
  return "Unavailable"
}

export function evalsReadinessReasonText(readiness: EvalsOperationReadiness): string {
  if (readiness.reason_code == null) {
    return "Available in this deployment; the API still enforces authorization and request preconditions."
  }
  return REASON_TEXT[readiness.reason_code]
}
