import { CapabilityRoute, useDashboardCapability } from "@/components/dashboard/server-contract"
import { SessionsPage } from "./sessions-page"

/** Session route capability is checked before its query hooks mount. */
export function SessionsEntry() {
  const execution = useDashboardCapability({ kind: "mutation", mutation: "session_execution" })
  return (
    <CapabilityRoute
      title="Sessions"
      requirement={{ kind: "surface", surface: "session_overview_queries" }}
    >
      <SessionsPage canStart={execution.enabled} />
    </CapabilityRoute>
  )
}
