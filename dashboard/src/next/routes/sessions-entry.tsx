import { CapabilityRoute, useDashboardCapability } from "@/components/dashboard/server-contract"
import { realApi } from "../api/real"
import { ServicesProvider } from "../app/services"
import { SessionsPage } from "./sessions-page"

/** Page-local services until the Home port installs the shared production provider. */
export function SessionsEntry() {
  const execution = useDashboardCapability({ kind: "mutation", mutation: "session_execution" })
  return (
    <CapabilityRoute
      title="Sessions"
      requirement={{ kind: "surface", surface: "session_overview_queries" }}
    >
      <ServicesProvider
        value={{ api: realApi, now: Date.now, executionEnabled: execution.enabled }}
      >
        <SessionsPage canStart={execution.enabled} />
      </ServicesProvider>
    </CapabilityRoute>
  )
}
