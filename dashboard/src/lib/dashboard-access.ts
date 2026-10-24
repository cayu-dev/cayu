import type { DashboardCapabilityRequirement } from "./dashboard-capabilities"
import type { AppInfoResponse, CapabilityOperation } from "./generated/server-api"

export const ROLE_FORBIDDEN_REASON =
  "Your role can read and decide, but this action requires full access."

export type DashboardOperation = Omit<CapabilityOperation, "unavailable_reason"> & {
  unavailable_reason?: CapabilityOperation["unavailable_reason"] | "role_forbidden"
}

export function parseDashboardIdentity(value: unknown): AppInfoResponse {
  if (
    value === null ||
    typeof value !== "object" ||
    !("name" in value) ||
    typeof value.name !== "string" ||
    !value.name.trim() ||
    !("runtime_version" in value) ||
    typeof value.runtime_version !== "string" ||
    !value.runtime_version.trim() ||
    !("role" in value) ||
    (value.role !== null && typeof value.role !== "string") ||
    !("full_access" in value) ||
    typeof value.full_access !== "boolean"
  )
    throw new Error("The server returned an invalid dashboard identity.")
  return value as AppInfoResponse
}

const MEMBER_READ_SURFACES = new Set([
  "app_info",
  "artifacts",
  "dashboard",
  "inbox",
  "inbox_queries",
  "pricing",
  "reviewed_knowledge",
  "session_follow",
  "session_trace",
  "session_overview",
  "session_overview_queries",
  "task_index",
  "task_creation",
  "session_display_metadata",
  "structured_output_transcript",
  "supervision",
  "tasks",
  "usage",
  "workflow",
])

/** UI admission only; the server independently authorizes every request. */
export function applyViewerAccess(
  operation: CapabilityOperation,
  requirement: DashboardCapabilityRequirement,
  identity: AppInfoResponse | null | undefined,
): DashboardOperation {
  if (identity?.full_access === true) return operation
  const authenticated =
    identity?.full_access === false && (typeof identity.role === "string" || identity.role === null)
  const permitted =
    authenticated &&
    (requirement.kind === "mutation"
      ? requirement.mutation === "pending_action_resolution" ||
        requirement.mutation === "knowledge_review"
      : (requirement.operation ?? "read") === "read" &&
        MEMBER_READ_SURFACES.has(requirement.surface))
  return permitted ? operation : { enabled: false, unavailable_reason: "role_forbidden" }
}

export function roleForbiddenMessage(status: number, detail: unknown): string | null {
  return status === 403 &&
    detail !== null &&
    typeof detail === "object" &&
    "code" in detail &&
    detail.code === "role_forbidden"
    ? ROLE_FORBIDDEN_REASON
    : null
}
