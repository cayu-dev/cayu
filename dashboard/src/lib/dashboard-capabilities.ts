import type {
  CapabilityOperation,
  ControlPlaneCapabilities,
  ControlPlaneMutationCapabilities,
  ControlPlaneSurfaceCapabilities,
} from "./generated/server-api"

export type DashboardSurface = keyof ControlPlaneSurfaceCapabilities
type DashboardSurfaceCapability = NonNullable<ControlPlaneSurfaceCapabilities[DashboardSurface]>
export type DashboardSurfaceOperation = keyof Pick<DashboardSurfaceCapability, "read" | "mutate">
export type DashboardMutation = keyof ControlPlaneMutationCapabilities

const UNSUPPORTED_OPTIONAL_SURFACE: CapabilityOperation = {
  enabled: false,
  unavailable_reason: "unsupported",
}

export type DashboardCapabilityRequirement =
  | {
      kind: "surface"
      surface: DashboardSurface
      operation?: DashboardSurfaceOperation
    }
  | {
      kind: "mutation"
      mutation: DashboardMutation
    }

export const DASHBOARD_ROUTE_REQUIREMENTS = {
  "/": { kind: "surface", surface: "dashboard" },
  "/tasks": { kind: "surface", surface: "tasks" },
  "/usage": { kind: "surface", surface: "usage" },
  "/knowledge": { kind: "surface", surface: "reviewed_knowledge" },
  "/artifacts": { kind: "surface", surface: "artifacts" },
  "/evals": { kind: "surface", surface: "evals" },
  "/sessions/$sessionId/workflow": { kind: "surface", surface: "workflow" },
  "/run": { kind: "mutation", mutation: "session_execution" },
} as const satisfies Record<string, DashboardCapabilityRequirement>

export function resolveDashboardCapability(
  capabilities: ControlPlaneCapabilities,
  requirement: DashboardCapabilityRequirement,
): CapabilityOperation {
  if (requirement.kind === "mutation") {
    return capabilities.mutations[requirement.mutation]
  }
  const surface = capabilities.surfaces[requirement.surface]
  return surface?.[requirement.operation ?? "read"] ?? UNSUPPORTED_OPTIONAL_SURFACE
}

export function dashboardCapabilityUnavailableText(operation: CapabilityOperation): string | null {
  if (operation.enabled) return null
  if (operation.unavailable_reason === "not_configured") {
    return "This feature is not configured for this Cayu deployment."
  }
  return "This operation is not supported by this Cayu deployment."
}

export function dashboardCapabilityEnabled(
  capabilities: ControlPlaneCapabilities,
  requirement: DashboardCapabilityRequirement,
): boolean {
  return resolveDashboardCapability(capabilities, requirement).enabled
}
