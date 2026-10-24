import { createContext, type ReactNode, useContext } from "react"
import type { ServerContract } from "../../lib/api"
import { applyViewerAccess } from "../../lib/dashboard-access"
import {
  type DashboardCapabilityRequirement,
  dashboardCapabilityUnavailableText,
  resolveDashboardCapability,
} from "../../lib/dashboard-capabilities"
import { PageBody, ShellContext, TopBar } from "../../next/routes/common"
import { Page, PageHeader, StateMessage } from "./layout"
import { useViewer } from "./viewer"

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
  return applyViewerAccess(
    resolveDashboardCapability(useServerContract().capabilities, requirement),
    requirement,
    useViewer(),
  )
}

export function CapabilityRoute({
  requirement,
  title,
  children,
  inline = false,
}: {
  requirement: DashboardCapabilityRequirement
  title: string
  children: ReactNode
  inline?: boolean
}) {
  const pageLayout = useContext(ShellContext)?.pageLayout
  const capability = useDashboardCapability(requirement)
  const unavailableText = dashboardCapabilityUnavailableText(capability)
  if (unavailableText === null) return children
  if (inline)
    return (
      <StateMessage
        data-testid="dashboard-capability-unavailable"
        className="rounded-lg border border-border bg-muted/30 py-12"
      >
        <div role="status">
          <div className="font-medium">{title} is unavailable</div>
          <div className="mt-1">{unavailableText}</div>
        </div>
      </StateMessage>
    )

  if (pageLayout)
    return (
      <>
        <TopBar>{title}</TopBar>
        <PageBody>
          <div
            data-testid="dashboard-capability-unavailable"
            role="status"
            className="rounded-lg border border-border bg-muted/30 p-6"
          >
            <div className="font-medium">{title} is unavailable</div>
            <div className="mt-1">{unavailableText}</div>
          </div>
        </PageBody>
      </>
    )

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
