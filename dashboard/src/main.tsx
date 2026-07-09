import "./index.css"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { createRootRoute, createRoute, createRouter, RouterProvider } from "@tanstack/react-router"
import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import { Layout } from "./Layout"
import { dashboardConfig } from "./lib/config"
import { DashboardPage } from "./routes/dashboard"
import { KnowledgePage } from "./routes/knowledge"
import { RunPage } from "./routes/run"
import { SessionDetailPage } from "./routes/session-detail"
import { SessionsPage } from "./routes/sessions"
import { TasksPage } from "./routes/tasks"
import { UsagePage } from "./routes/usage"

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

const sessionDetailRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/sessions/$sessionId",
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
  sessionDetailRoute,
  knowledgeRoute,
  runRoute,
])

const router = createRouter({
  routeTree,
  basepath: dashboardConfig.basePath === "/" ? undefined : dashboardConfig.basePath,
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
