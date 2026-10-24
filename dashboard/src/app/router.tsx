import {
  createRootRoute,
  createRoute,
  createRouter,
  lazyRouteComponent,
  type RouterHistory,
  redirect,
} from "@tanstack/react-router"
import { CapabilityRoute } from "../components/dashboard/server-contract"
import { FullAccessRoute } from "../components/dashboard/viewer"
import { dashboardConfig } from "../lib/config"
import { DASHBOARD_ROUTE_REQUIREMENTS } from "../lib/dashboard-capabilities"
import { validateEvalsSearch } from "../lib/evals-search"
import { parseDashboardSearch, stringifyDashboardSearch } from "../lib/search-params"
import { validateSessionHistorySearch } from "../lib/session-history-search"
import { validateSessionIndexSearch } from "../lib/session-index-search"
import { validateUsageRollupSearch } from "../lib/usage-rollup-search"
import { validateWorkflowSearch } from "../lib/workflow-search"
import { Layout } from "./layout"

const AgentsPage = lazyRouteComponent(() => import("../routes/agents"), "AgentsPage")
const ArtifactsPage = lazyRouteComponent(() => import("../routes/artifacts"), "ArtifactsPage")
const DashboardPage = lazyRouteComponent(() => import("../next/routes/home-page"), "HomePage")
const EnvironmentsPage = lazyRouteComponent(
  () => import("../routes/environments"),
  "EnvironmentsPage",
)
const EvalsPage = lazyRouteComponent(() => import("../routes/evals"), "EvalsPage")
const KnowledgePage = lazyRouteComponent(() => import("../routes/knowledge"), "KnowledgePage")
const PendingActionsPage = lazyRouteComponent(
  () => import("../next/routes/inbox-page"),
  "InboxPage",
)
const SessionDetailPage = lazyRouteComponent(
  () => import("../routes/session-detail"),
  "SessionDetailPage",
)
const SessionsPage = lazyRouteComponent(
  () => import("../next/routes/sessions-entry"),
  "SessionsEntry",
)
const TasksPage = lazyRouteComponent(() => import("../routes/tasks"), "TasksPage")
const UsagePage = lazyRouteComponent(() => import("../routes/usage"), "UsagePage")
const WorkflowPage = lazyRouteComponent(() => import("../routes/workflow"), "WorkflowPage")
const SystemPage = lazyRouteComponent(() => import("../routes/system"), "SystemPage")

function RoutePending() {
  return (
    <div className="space-y-4" role="status" aria-live="polite">
      <div className="h-8 w-48 animate-pulse rounded bg-muted" />
      <div className="h-32 animate-pulse rounded-lg border border-border bg-muted/40" />
      <span className="sr-only">Loading page</span>
    </div>
  )
}

export function createDashboardRouter(history?: RouterHistory) {
  const rootRoute = createRootRoute({ component: Layout })

  const dashboardRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: "/",
    validateSearch: (search: Record<string, unknown>): { start?: string } => ({
      start: search.start == null ? undefined : String(search.start),
    }),
    component: DashboardPage,
  })

  const sessionsRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: "/sessions",
    validateSearch: validateSessionIndexSearch,
    component: SessionsPage,
  })

  const usageRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: "/usage",
    validateSearch: validateUsageRollupSearch,
    component: () => (
      <CapabilityRoute requirement={DASHBOARD_ROUTE_REQUIREMENTS["/usage"]} title="Usage">
        <UsagePage />
      </CapabilityRoute>
    ),
  })

  const evalsRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: "/evals",
    validateSearch: validateEvalsSearch,
    component: () => (
      <FullAccessRoute title="Evals">
        <CapabilityRoute requirement={DASHBOARD_ROUTE_REQUIREMENTS["/evals"]} title="Evals">
          <EvalsPage />
        </CapabilityRoute>
      </FullAccessRoute>
    ),
  })

  const tasksRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: "/tasks",
    component: () => (
      <CapabilityRoute requirement={DASHBOARD_ROUTE_REQUIREMENTS["/tasks"]} title="Tasks">
        <TasksPage />
      </CapabilityRoute>
    ),
  })

  const pendingActionsRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: "/inbox",
    validateSearch: (
      search: Record<string, unknown>,
    ): { kind?: string; q?: string; session_id?: string; agent_name?: string } => ({
      kind: typeof search.kind === "string" ? search.kind : undefined,
      q: typeof search.q === "string" ? search.q : undefined,
      session_id: typeof search.session_id === "string" ? search.session_id : undefined,
      agent_name: typeof search.agent_name === "string" ? search.agent_name : undefined,
    }),
    component: () => (
      <CapabilityRoute requirement={{ kind: "surface", surface: "inbox_queries" }} title="Inbox">
        <PendingActionsPage />
      </CapabilityRoute>
    ),
  })

  const agentsRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: "/agents",
    component: AgentsPage,
  })

  const environmentsRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: "/environments",
    component: EnvironmentsPage,
  })

  const artifactsRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: "/artifacts",
    component: () => (
      <CapabilityRoute requirement={DASHBOARD_ROUTE_REQUIREMENTS["/artifacts"]} title="Artifacts">
        <ArtifactsPage />
      </CapabilityRoute>
    ),
  })

  const sessionDetailRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: "/sessions/$sessionId",
    validateSearch: validateSessionHistorySearch,
    remountDeps: ({ params }) => ({ sessionId: params.sessionId }),
    component: SessionDetailPage,
  })

  const workflowRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: "/sessions/$sessionId/workflow",
    validateSearch: validateWorkflowSearch,
    remountDeps: ({ params }) => ({ sessionId: params.sessionId }),
    component: () => (
      <CapabilityRoute
        requirement={DASHBOARD_ROUTE_REQUIREMENTS["/sessions/$sessionId/workflow"]}
        title="Workflow"
      >
        <WorkflowPage />
      </CapabilityRoute>
    ),
  })

  const knowledgeRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: "/knowledge",
    component: () => (
      <CapabilityRoute requirement={DASHBOARD_ROUTE_REQUIREMENTS["/knowledge"]} title="Knowledge">
        <KnowledgePage />
      </CapabilityRoute>
    ),
  })

  const runRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: "/run",
    beforeLoad: () => {
      throw redirect({ to: "/", search: { start: "1" } })
    },
  })
  const oldPendingRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: "/pending-actions",
    beforeLoad: ({ search }) => {
      throw redirect({ to: "/inbox", search })
    },
  })

  const systemRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: "/system",
    component: () => (
      <FullAccessRoute title="System">
        <SystemPage />
      </FullAccessRoute>
    ),
  })

  const routeTree = rootRoute.addChildren([
    dashboardRoute,
    sessionsRoute,
    evalsRoute,
    usageRoute,
    tasksRoute,
    pendingActionsRoute,
    oldPendingRoute,
    agentsRoute,
    environmentsRoute,
    artifactsRoute,
    workflowRoute,
    sessionDetailRoute,
    knowledgeRoute,
    runRoute,
    systemRoute,
  ])

  return createRouter({
    history,
    routeTree,
    basepath: dashboardConfig.basePath === "/" ? undefined : dashboardConfig.basePath,
    parseSearch: parseDashboardSearch,
    stringifySearch: stringifyDashboardSearch,
    defaultPendingComponent: RoutePending,
    defaultPreload: "intent",
  })
}
