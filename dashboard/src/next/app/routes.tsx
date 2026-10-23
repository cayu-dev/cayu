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
import { EvalRunPage } from "../routes/eval-run-page"
import { EvalSettingsPage } from "../routes/eval-settings-page"
import { EvalSuitePage } from "../routes/eval-suite-page"
import { EvalsPage } from "../routes/evals-page"
import { HomePage } from "../routes/home-page"
import { InboxPage } from "../routes/inbox-page"
import { KnowledgePage } from "../routes/knowledge-page"
import { Layout } from "../routes/layout"
import { LegacyPage } from "../routes/legacy-page"
import { SessionPage } from "../routes/session/session-page"
import { SessionsPage } from "../routes/sessions-page"
import { SpecialistsPage } from "../routes/specialists-page"
import { TaskPage } from "../routes/task-page"
import { TasksPage } from "../routes/tasks-page"
import { ToolsPage } from "../routes/tools-page"

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
  const moved = (path: string, to: string) =>
    createRoute({
      getParentRoute: () => root,
      path,
      beforeLoad: () => {
        throw redirect({ to: to as never })
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
      validateSearch: (s: Record<string, unknown>) => ({
        q: str(s.q),
        state: str(s.state),
        agent: str(s.agent),
        label: str(s.label),
      }),
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
          to: "/sessions/$sessionId" as never,
          params: params as never,
          search: { view: "trace" } as never,
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
    page("/specialists", SpecialistsPage),
    page("/tools", ToolsPage),
    page("/knowledge", KnowledgePage),
    ...(
      [
        ["/usage", "Usage"],
        ["/environments", "Environments"],
        ["/artifacts", "Artifacts"],
        ["/system", "System"],
      ] as const
    ).map(([path, title]) => page(path, () => <LegacyPage title={title} />)),
    moved("/pending-actions", "/inbox"),
    moved("/run", "/"),
    moved("/agents", "/specialists"),
  ])

  return createRouter({ routeTree: tree, history, defaultPreload: "intent" })
}
