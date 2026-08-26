import type { EvalsOperationReadiness, EvalsReadiness } from "./generated/server-api"

type EvalsReadinessReasonCode = NonNullable<EvalsOperationReadiness["reason_code"]>

export const EVALS_READINESS_OPERATIONS = [
  ["captured_evaluation", "Captured evaluation"],
  ["catalog_read", "Catalog"],
  ["catalog_write", "Import and save"],
  ["captured_result_persistence", "Captured results"],
  ["scenario_conversion", "Production scenarios"],
  ["fresh_launch", "Current-app trials"],
  ["cancellation", "Cancellation"],
  ["comparison", "Comparison"],
  ["reports", "Reports"],
] as const satisfies ReadonlyArray<readonly [keyof EvalsReadiness, string]>

const REASON_TEXT: Record<EvalsReadinessReasonCode, string> = {
  evaluation_promotion_not_configured:
    "The optional runnable-session conversion adapter is not configured in this deployment.",
  terminal_evidence_not_supported:
    "The session store cannot provide the terminal evidence required for captured evaluation.",
  session_lineage_not_supported:
    "The session store cannot provide the session lineage required for captured evaluation.",
  eval_store_not_configured:
    "For mount_cayu, pass access=AuthenticatedAccess(...) and complete durable Evals wiring through evals=EvalsConfig(target=target, store=eval_store), where eval_store is a SQLiteEvalStore or PostgresEvalStore.",
  eval_target_not_configured:
    "For mount_cayu, pass access=AuthenticatedAccess(...) and complete durable Evals wiring through evals=EvalsConfig(target=target, store=eval_store), where target is a CorpusTarget for the exact mounted CayuApp.",
  captured_result_persistence_not_available:
    "The configured eval store cannot persist captured results.",
  scenario_v2_not_available: "Scenario-v2 conversion is unavailable in this deployment.",
}

export function evalsReadinessStateLabel(readiness: EvalsOperationReadiness): string {
  if (readiness.state === "ready") return "Ready"
  if (readiness.state === "gated") return "Not ready"
  return "Unavailable"
}

export function evalsReadinessReasonText(readiness: EvalsOperationReadiness): string {
  if (readiness.reason_code == null) {
    return "Available in this deployment; the API still enforces authorization and request preconditions."
  }
  return REASON_TEXT[readiness.reason_code]
}
