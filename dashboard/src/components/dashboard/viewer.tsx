import { createContext, type ReactNode, useContext } from "react"
import { ROLE_FORBIDDEN_REASON } from "../../lib/dashboard-access"
import type { AppInfoResponse } from "../../lib/generated/server-api"
import { PageBody, ShellContext, TopBar } from "../../next/routes/common"
import { Page, PageHeader, StateMessage } from "./layout"

const ViewerContext = createContext<AppInfoResponse | null>(null)
export const ViewerProvider = ViewerContext.Provider
export function useViewer() {
  return useContext(ViewerContext)
}

export function FullAccessRoute({ title, children }: { title: string; children: ReactNode }) {
  const viewer = useViewer()
  const pageLayout = useContext(ShellContext)?.pageLayout
  if (viewer?.full_access === true) return children
  if (pageLayout)
    return (
      <>
        <TopBar>{title}</TopBar>
        <PageBody>
          <div
            data-testid="dashboard-role-unavailable"
            role="status"
            className="rounded-lg border border-border bg-muted/30 p-6"
          >
            {ROLE_FORBIDDEN_REASON}
          </div>
        </PageBody>
      </>
    )
  return (
    <Page data-testid="dashboard-role-unavailable">
      <PageHeader title={title} />
      <StateMessage role="status">{ROLE_FORBIDDEN_REASON}</StateMessage>
    </Page>
  )
}
