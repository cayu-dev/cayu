/**
 * Plain names and one-line reasons for the System page's "What's turned off" list. Keys
 * follow the server's capability and Evals readiness fields; anything new falls back to a
 * readable version of its key.
 */
import type {
  CapabilityOperation,
  ControlPlaneMutationCapabilities,
  ControlPlaneSurfaceCapabilities,
  EvalsOperationReadiness,
  EvalsReadiness,
  SystemDiagnosticsResponse,
} from "@/lib/generated/server-api"

export const readable = (key: string) =>
  key.charAt(0).toUpperCase() + key.slice(1).replaceAll("_", " ")

const SURFACE: Partial<Record<keyof ControlPlaneSurfaceCapabilities, string>> = {
  app_info: "Agent name and your role",
  artifacts: "Artifacts",
  browser_recordings: "Browser recordings",
  dashboard: "Dashboard",
  eval_reported_views: "Eval reports",
  eval_suite_views: "Eval suite pages",
  evals: "Evals",
  evaluation_promotion: "Turning sessions into eval cases",
  inbox: "Inbox",
  inbox_queries: "Inbox search and counts",
  kept_knowledge: "Kept knowledge",
  pricing: "Cost estimates",
  reviewed_knowledge: "Knowledge review",
  session_display_metadata: "Agent-set session titles",
  session_follow: "Live session updates",
  session_followups: "Follow-up messages",
  session_overview: "Session summaries",
  session_overview_queries: "Session counts",
  session_trace: "Session trace",
  structured_output_transcript: "Structured replies in transcripts",
  supervision: "Supervision",
  task_creation: "Creating tasks",
  task_graph_creation: "Task graph creation",
  task_index: "Task filters and counts",
  tasks: "Tasks",
  tool_display: "Tool display names",
  tool_inventory: "Tool list",
  tool_usage: "Tool usage",
  usage: "Usage",
  workflow: "Workflow view",
}

/** The graph-creation surface intentionally has no read operation. */
const PRIMARY_OPERATION: Partial<Record<keyof ControlPlaneSurfaceCapabilities, "mutate">> = {
  task_graph_creation: "mutate",
}

/** Surfaces whose writes are a feature people use, so a read-only state is worth saying. */
const WRITABLE = new Set<string>([
  "inbox",
  "tasks",
  "task_creation",
  "reviewed_knowledge",
  "kept_knowledge",
  "evals",
])

const NOT_SET_UP: Partial<Record<keyof ControlPlaneSurfaceCapabilities, string>> = {
  browser_recordings: "No place to store browser recordings is set up.",
  evaluation_promotion: "No eval store is set up to receive cases.",
  pricing: "No price list is set up.",
  evals: "No eval store is set up.",
  artifacts: "No artifact store is set up.",
  tasks: "No task store is set up.",
  inbox: "No inbox is set up.",
}

const MUTATION: Record<keyof ControlPlaneMutationCapabilities, string> = {
  session_execution: "Starting sessions and replying to them",
  session_interruption: "Stopping running sessions",
  provider_operation_resolution: "Resolving model calls that stopped partway",
  pending_action_resolution: "Approving and answering from the inbox",
  session_annotations: "Notes on sessions",
  task_lifecycle: "Pausing, resuming and cancelling tasks",
  knowledge_review: "Keeping or dropping knowledge",
}

function reason(op: CapabilityOperation, notSetUp?: string) {
  return op.unavailable_reason === "not_configured"
    ? (notSetUp ?? "Not set up on this deployment.")
    : "This deployment doesn't support it."
}

export type Off = { key: string; name: string; why: string }

export function turnedOff(d: SystemDiagnosticsResponse): Off[] {
  const off: Off[] = []
  for (const [key, surface] of Object.entries(d.capabilities.surfaces)) {
    if (!surface) continue
    const name = SURFACE[key as keyof ControlPlaneSurfaceCapabilities] ?? readable(key)
    const notSetUp = NOT_SET_UP[key as keyof ControlPlaneSurfaceCapabilities]
    const primary =
      surface[PRIMARY_OPERATION[key as keyof ControlPlaneSurfaceCapabilities] ?? "read"]
    if (!primary.enabled) off.push({ key: `surface.${key}`, name, why: reason(primary, notSetUp) })
    else if (WRITABLE.has(key) && !surface.mutate.enabled)
      off.push({
        key: `surface.${key}.mutate`,
        name: `${name} (read-only)`,
        why: `You can look but not change anything. ${reason(surface.mutate, notSetUp)}`,
      })
  }
  for (const [key, op] of Object.entries(d.capabilities.mutations)) {
    if (op.enabled) continue
    off.push({
      key: `mutation.${key}`,
      name: MUTATION[key as keyof ControlPlaneMutationCapabilities] ?? readable(key),
      why: reason(op),
    })
  }
  return off
}

export const EVAL_OPERATIONS: [keyof EvalsReadiness, string][] = [
  ["catalog_read", "Seeing suites"],
  ["catalog_write", "Creating and editing suites"],
  ["fresh_launch", "Running suites against the agent"],
  ["cancellation", "Stopping runs"],
  ["reports", "Run reports"],
  ["comparison", "Comparing runs"],
  ["captured_evaluation", "Scoring past sessions"],
  ["captured_result_persistence", "Saving scores for past sessions"],
  ["session_additions", "Adding sessions to a suite"],
  ["scenario_conversion", "Turning sessions into scenarios"],
]

const EVAL_REASON: Record<NonNullable<EvalsOperationReadiness["reason_code"]>, string> = {
  evaluation_promotion_not_configured: "Turning sessions into eval cases isn't set up.",
  terminal_evidence_not_supported: "The session store can't record how sessions ended.",
  session_lineage_not_supported: "The session store can't link sessions to the runs behind them.",
  eval_store_not_configured: "No eval store is set up.",
  eval_target_not_configured: "No agent is set up for evals to run against.",
  captured_result_persistence_not_available: "The eval store can't save scores for past sessions.",
  session_additions_not_available: "The eval store can't add sessions to existing suites.",
  scenario_v2_not_available: "The eval store is too old to turn sessions into scenarios.",
}

export function evalReason(r: EvalsOperationReadiness): string {
  if (r.reason_code) return EVAL_REASON[r.reason_code] ?? readable(r.reason_code)
  return r.state === "gated" ? "Needs setup." : "This deployment doesn't support it."
}

export const STORE: Record<string, string> = {
  session: "Sessions",
  task: "Tasks",
  knowledge: "Knowledge",
  artifact: "Artifacts",
}
