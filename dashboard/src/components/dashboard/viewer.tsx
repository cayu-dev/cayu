import { createContext, type ReactNode, useContext } from "react"
import { ROLE_FORBIDDEN_REASON } from "../../lib/dashboard-access"
import type { AppInfoResponse } from "../../lib/generated/server-api"
import { Page, PageHeader, StateMessage } from "./layout"

const ViewerContext = createContext<AppInfoResponse | null>(null)
export const ViewerProvider = ViewerContext.Provider
export function useViewer() {
  return useContext(ViewerContext)
}

export function FullAccessRoute({ title, children }: { title: string; children: ReactNode }) {
  const viewer = useViewer()
  if (viewer?.full_access === true) return children
  return (
    <Page data-testid="dashboard-role-unavailable">
      <PageHeader title={title} />
      <StateMessage role="status">{ROLE_FORBIDDEN_REASON}</StateMessage>
    </Page>
  )
}
