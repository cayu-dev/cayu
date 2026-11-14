/**
 * The Developer dashboard as the host of its own parts (src/parts): its API client, its
 * router's links, the server contract's capabilities, the viewer's access and protected
 * human review. Rendered once inside the services, so every page and the shell read the
 * same host.
 */
import { useQueryClient } from "@tanstack/react-query"
import { type ReactNode, useMemo } from "react"
import { useViewer } from "@/components/dashboard/viewer"
import { fetchPendingKnowledgeEntry } from "@/lib/api"
import { dashboardConfig } from "@/lib/config"
import { dashboardReviewApi } from "@/lib/human-review"
import { fetchSessionRecovery } from "@/lib/session-recovery"
import {
  type InboxApi,
  type PartLink,
  type PartsHost,
  PartsHostProvider,
  type ProtectedReview,
  ProtectedReviewPart,
} from "../../parts"
import { NavLink, sessionRoute, taskRoute } from "../routes/common"
import { useServices } from "./services"

const DashboardLink: PartLink = ({ to, className, children }) =>
  "session" in to ? (
    <NavLink {...sessionRoute(to.session)} className={className}>
      {children}
    </NavLink>
  ) : (
    <NavLink {...taskRoute(to.task)} className={className}>
      {children}
    </NavLink>
  )

const purpose = dashboardConfig.humanReviewPurpose
const DashboardProtectedReview: ProtectedReview | undefined = purpose
  ? (props) => <ProtectedReviewPart {...props} api={dashboardReviewApi} purpose={purpose} />
  : undefined

/** What else a decision can change on the dashboard's pages, beyond the Inbox itself. */
const SETTLED_KEYS = [
  "child-actions",
  "home",
  "sessions",
  "session-list",
  "overview",
  "transcript",
  "knowledge",
  "tool-usage",
]

export function DashboardPartsHost({ children }: { children: ReactNode }) {
  const services = useServices()
  const viewer = useViewer()
  const queryClient = useQueryClient()
  const { api, now, prototype, inboxEnabled, decisionsEnabled, knowledgeReviewEnabled } = services
  const fullAccess = viewer?.full_access === true
  const host = useMemo<PartsHost>(() => {
    const inboxApi: InboxApi = {
      list: (query, signal) => api.inbox(query, signal),
      counts: (query, signal) => api.inboxCounts(query, signal),
      settle: (id, body) => api.settle(id, body),
      sessionTitles: (ids, signal) => api.overviews(ids, signal),
    }
    return {
      now,
      Link: DashboardLink,
      onSettled: async () => {
        await Promise.all(
          SETTLED_KEYS.map((key) => queryClient.invalidateQueries({ queryKey: [key] })),
        )
      },
      // The prototype's fake data allows every action and settles recoveries in place.
      inbox: prototype
        ? {
            api: { ...inboxApi, recover: (sessionId, body) => api.recover(sessionId, body) },
            enabled: true,
            decide: true,
            reviewKnowledge: true,
            fullAccess: true,
            ProtectedReview: DashboardProtectedReview,
          }
        : {
            api: {
              ...inboxApi,
              knowledgeEntry: fetchPendingKnowledgeEntry,
              sessionRecovery: fetchSessionRecovery,
            },
            enabled: inboxEnabled === true,
            decide: decisionsEnabled === true,
            reviewKnowledge: knowledgeReviewEnabled === true,
            fullAccess,
            ProtectedReview: DashboardProtectedReview,
          },
    }
  }, [
    api,
    now,
    prototype,
    inboxEnabled,
    decisionsEnabled,
    knowledgeReviewEnabled,
    fullAccess,
    queryClient,
  ])
  return <PartsHostProvider value={host}>{children}</PartsHostProvider>
}
