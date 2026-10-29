/**
 * The reworked dashboard's routes. Old URLs keep working: /pending-actions opens the inbox, /run opens Home, /agents opens Specialists, and
 * /sessions/:id/workflow opens the session's trace.
 */
import {
  createRootRoute,
  createRoute,
  createRouter,
  type RouterHistory,
  redirect,
} from "@tanstack/react-router"
import { validateArtifactSearch } from "@/lib/artifact-search"
import { validateSessionIndexSearch } from "@/lib/session-index-search"
import { validateUsageRollupSearch } from "@/lib/usage-rollup-search"
import { ArtifactsPage } from "../routes/artifacts-page"
import { EnvironmentPage, EnvironmentsPage } from "../routes/environments-page"
import { EvalRunPage } from "../routes/eval-run-page"
import { EvalSettingsPage } from "../routes/eval-settings-page"
import { EvalSuitePage } from "../routes/eval-suite-page"
import { EvalsPage } from "../routes/evals-page"
import { HomePage } from "../routes/home-page"
import { InboxPage } from "../routes/inbox-page"
import { KnowledgePage } from "../routes/knowledge-page"
import { Layout } from "../routes/layout"
import { SessionPage } from "../routes/session/session-page"
import { SessionsPage } from "../routes/sessions-page"
import { SpecialistsPage } from "../routes/specialists-page"
import { SystemPage } from "../routes/system-page"
import { TaskPage } from "../routes/task-page"
import { TasksPage } from "../routes/tasks-page"
import { ToolsPage } from "../routes/tools-page"
import { UsagePage } from "../routes/usage-page"

// The default search parser turns "1" into a number and "true" into a boolean.
const str = (v: unknown) =>
  typeof v === "string" && v
    ? v
    : typeof v === "number" || typeof v === "boolean"
      ? String(v)
      : undefined

export function createDashboardRouter(history?: RouterHistory) {
  const root = createRootRoute({ component: Layout })
  const page = (path: string, component: () => React.ReactNode) =>
    createRoute({ getParentRoute: () => root, path, component })
  const moved = (path: string, to: "/" | "/inbox" | "/specialists") =>
    createRoute({
      getParentRoute: () => root,
      path,
      beforeLoad: () => {
        throw redirect({ to })
      },
    })

  const tree = root.addChildren([
    createRoute({
      getParentRoute: () => root,
      path: "/",
      validateSearch: (s: Record<string, unknown>) => ({ start: str(s.start) }),
      component: HomePage,
    }),
    createRoute({
      getParentRoute: () => root,
      path: "/sessions",
      validateSearch: validateSessionIndexSearch,
      component: SessionsPage,
    }),
    createRoute({
      getParentRoute: () => root,
      path: "/sessions/$sessionId",
      validateSearch: (s: Record<string, unknown>) => ({
        panel: str(s.panel),
        view: str(s.view),
        span: str(s.span),
        event: str(s.event),
        log: str(s.log),
      }),
      component: SessionPage,
    }),
    createRoute({
      getParentRoute: () => root,
      path: "/sessions/$sessionId/workflow",
      beforeLoad: ({ params }) => {
        throw redirect({
          to: "/sessions/$sessionId",
          params,
          search: { view: "trace" },
        })
      },
    }),
    createRoute({
      getParentRoute: () => root,
      path: "/inbox",
      validateSearch: (s: Record<string, unknown>) => ({ kind: str(s.kind) }),
      component: InboxPage,
    }),
    createRoute({
      getParentRoute: () => root,
      path: "/tasks",
      validateSearch: (s: Record<string, unknown>) => ({
        group: str(s.group),
        q: str(s.q),
        type: str(s.type),
        agent: str(s.agent),
        cayu: str(s.cayu),
      }),
      component: TasksPage,
    }),
    page("/tasks/$taskId", TaskPage),
    page("/evals", EvalsPage),
    page("/evals/settings", EvalSettingsPage),
    page("/evals/suites/$suiteId", EvalSuitePage),
    createRoute({
      getParentRoute: () => root,
      path: "/evals/runs/$runId",
      validateSearch: (s: Record<string, unknown>) => ({
        compare: str(s.compare),
        filter: str(s.filter),
        case: str(s.case),
      }),
      component: EvalRunPage,
    }),
    createRoute({
      getParentRoute: () => root,
      path: "/specialists",
      validateSearch: (s: Record<string, unknown>) => ({ q: str(s.q) }),
      component: SpecialistsPage,
    }),
    createRoute({
      getParentRoute: () => root,
      path: "/tools",
      validateSearch: (s: Record<string, unknown>) => ({ q: str(s.q) }),
      component: ToolsPage,
    }),
    page("/knowledge", KnowledgePage),
    createRoute({
      getParentRoute: () => root,
      path: "/usage",
      validateSearch: validateUsageRollupSearch,
      component: UsagePage,
    }),
    createRoute({
      getParentRoute: () => root,
      path: "/environments",
      validateSearch: (s: Record<string, unknown>) => ({ q: str(s.q) }),
      component: EnvironmentsPage,
    }),
    page("/environments/$name", EnvironmentPage),
    createRoute({
      getParentRoute: () => root,
      path: "/artifacts",
      validateSearch: validateArtifactSearch,
      component: ArtifactsPage,
    }),
    page("/system", SystemPage),
    moved("/pending-actions", "/inbox"),
    moved("/run", "/"),
    moved("/agents", "/specialists"),
  ])

  return createRouter({ routeTree: tree, history, defaultPreload: "intent" })
}
