import "./index.css"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import {
  createRootRoute,
  createRoute,
  createRouter,
  lazyRouteComponent,
  RouterProvider,
} from "@tanstack/react-router"
import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import { Layout } from "./Layout"
import { dashboardConfig } from "./lib/config"
import { parseDashboardSearch, stringifyDashboardSearch } from "./lib/search-params"
import { validateSessionHistorySearch } from "./lib/session-history-search"
import { validateSessionIndexSearch } from "./lib/session-index-search"

const AgentsPage = lazyRouteComponent(() => import("./routes/agents"), "AgentsPage")
const ArtifactsPage = lazyRouteComponent(() => import("./routes/artifacts"), "ArtifactsPage")
const DashboardPage = lazyRouteComponent(() => import("./routes/dashboard"), "DashboardPage")
const EnvironmentsPage = lazyRouteComponent(
  () => import("./routes/environments"),
  "EnvironmentsPage",
)
const KnowledgePage = lazyRouteComponent(() => import("./routes/knowledge"), "KnowledgePage")
const PendingActionsPage = lazyRouteComponent(
  () => import("./routes/pending-actions"),
  "PendingActionsPage",
)
const RunPage = lazyRouteComponent(() => import("./routes/run"), "RunPage")
const SessionDetailPage = lazyRouteComponent(
  () => import("./routes/session-detail"),
  "SessionDetailPage",
)
const SessionsPage = lazyRouteComponent(() => import("./routes/sessions"), "SessionsPage")
const TasksPage = lazyRouteComponent(() => import("./routes/tasks"), "TasksPage")
const UsagePage = lazyRouteComponent(() => import("./routes/usage"), "UsagePage")

function RoutePending() {
  return (
    <div className="space-y-4" role="status" aria-live="polite">
      <div className="h-8 w-48 animate-pulse rounded bg-muted" />
      <div className="h-32 animate-pulse rounded-lg border border-border bg-muted/40" />
      <span className="sr-only">Loading page</span>
    </div>
  )
}

const queryClient = new QueryClient()

const rootRoute = createRootRoute({ component: Layout })

const dashboardRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
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
  component: UsagePage,
})

const tasksRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/tasks",
  component: TasksPage,
})

const pendingActionsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/pending-actions",
  component: PendingActionsPage,
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
  component: ArtifactsPage,
})

const sessionDetailRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/sessions/$sessionId",
  validateSearch: validateSessionHistorySearch,
  remountDeps: ({ params }) => ({ sessionId: params.sessionId }),
  component: SessionDetailPage,
})

const knowledgeRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/knowledge",
  component: KnowledgePage,
})

const runRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/run",
  component: RunPage,
})

const routeTree = rootRoute.addChildren([
  dashboardRoute,
  sessionsRoute,
  usageRoute,
  tasksRoute,
  pendingActionsRoute,
  agentsRoute,
  environmentsRoute,
  artifactsRoute,
  sessionDetailRoute,
  knowledgeRoute,
  runRoute,
])

const router = createRouter({
  routeTree,
  basepath: dashboardConfig.basePath === "/" ? undefined : dashboardConfig.basePath,
  parseSearch: parseDashboardSearch,
  stringifySearch: stringifyDashboardSearch,
  defaultPendingComponent: RoutePending,
  defaultPreload: "intent",
})

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router
  }
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  </StrictMode>,
)
