import type { SessionState } from "./api"

export function providerOperationNeedsResolution(
  operation: SessionState["provider_operation"] | null | undefined,
): boolean {
  return (
    (operation?.status === "provider_operation_unavailable" ||
      operation?.status === "ambiguous_submission") &&
    operation.resolution_action == null &&
    (operation.allowed_resolutions?.length ?? 0) > 0
  )
}
