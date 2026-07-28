import { createContext, type ReactNode, useContext } from "react"
import type { ServerContract } from "../../lib/api"
import {
  type DashboardCapabilityRequirement,
  dashboardCapabilityUnavailableText,
  resolveDashboardCapability,
} from "../../lib/dashboard-capabilities"
import { Page, PageHeader, StateMessage } from "./layout"

const ServerContractContext = createContext<ServerContract | null>(null)

export function ServerContractProvider({
  contract,
  children,
}: {
  contract: ServerContract
  children: ReactNode
}) {
  return (
    <ServerContractContext.Provider value={contract}>{children}</ServerContractContext.Provider>
  )
}

export function useServerContract(): ServerContract {
  const contract = useContext(ServerContractContext)
  if (contract === null) {
    throw new Error("The dashboard server contract is unavailable outside its compatibility gate.")
  }
  return contract
}

export function useDashboardCapability(requirement: DashboardCapabilityRequirement) {
  return resolveDashboardCapability(useServerContract().capabilities, requirement)
}

export function CapabilityRoute({
  requirement,
  title,
  children,
}: {
  requirement: DashboardCapabilityRequirement
  title: string
  children: ReactNode
}) {
  const capability = useDashboardCapability(requirement)
  const unavailableText = dashboardCapabilityUnavailableText(capability)
  if (unavailableText === null) return children

  return (
    <Page data-testid="dashboard-capability-unavailable">
      <PageHeader title={title} />
      <StateMessage className="rounded-lg border border-border bg-muted/30 py-12">
        <div role="status">
          <div className="font-medium">{title} is unavailable</div>
          <div className="mt-1">{unavailableText}</div>
        </div>
      </StateMessage>
    </Page>
  )
}
