import "./route-layout"
import {
  createRootRoute,
  createRoute,
  createRouter,
  lazyRouteComponent,
  type RouterHistory,
  redirect,
  type SearchSchemaInput,
} from "@tanstack/react-router"
import { useContext } from "react"
import { CapabilityRoute } from "../components/dashboard/server-contract"
import { FullAccessRoute } from "../components/dashboard/viewer"
import { validateArtifactSearch } from "../lib/artifact-search"
import { dashboardConfig } from "../lib/config"
import { DASHBOARD_ROUTE_REQUIREMENTS } from "../lib/dashboard-capabilities"
import { type EvalSettingsTool, validateEvalSettingsSearch } from "../lib/eval-settings-search"
import { validateEvalsSearch } from "../lib/evals-search"
import { parseDashboardSearch, stringifyDashboardSearch } from "../lib/search-params"
import { validateSessionDetailSearch, workflowSearchPatch } from "../lib/session-detail-search"
import { validateSessionIndexSearch } from "../lib/session-index-search"
import { validateUsageRollupSearch } from "../lib/usage-rollup-search"
import { validateWorkflowSearch } from "../lib/workflow-search"
import {
  evalCaseFilter,
  nativeRunSelector,
  resultSelector,
  suiteSelector,
} from "../next/app/eval-view-model"
import { PageBody, ShellContext, TopBar } from "../next/routes/common"
import { Layout } from "./layout"

const SpecialistsPage = lazyRouteComponent(
  () => import("../next/routes/specialists-entry"),
  "SpecialistsEntry",
)
const ToolsPage = lazyRouteComponent(() => import("../next/routes/tools-entry"), "ToolsEntry")
const ArtifactsPage = lazyRouteComponent(
  () => import("../next/routes/artifacts-page"),
  "ArtifactsPage",
)
const DashboardPage = lazyRouteComponent(() => import("../next/routes/home-page"), "HomePage")
const EnvironmentsPage = lazyRouteComponent(
  () => import("../next/routes/environments-page"),
  "EnvironmentsPage",
)
const EnvironmentPage = lazyRouteComponent(
  () => import("../next/routes/environments-page"),
  "EnvironmentPage",
)
const EvalsPage = lazyRouteComponent(
  () => import("../next/routes/evals/index-page"),
  "EvalsIndexPage",
)
const EvalSuitePage = lazyRouteComponent(
  () => import("../next/routes/evals/suite-page"),
  "EvalSuiteViewPage",
)
const EvalSuiteEditorPage = lazyRouteComponent(
  () => import("../next/routes/evals/editor-page"),
  "EvalSuiteEditorPage",
)
const EvalRunPage = lazyRouteComponent(
  () => import("../next/routes/evals/run-page"),
  "EvalRunViewPage",
)
const EvalSettingsPage = lazyRouteComponent(
  () => import("../next/routes/evals/settings-page"),
  "EvalSettingsViewPage",
)
const KnowledgePage = lazyRouteComponent(
  () => import("../next/routes/knowledge-entry"),
  "KnowledgeEntry",
)
const PendingActionsPage = lazyRouteComponent(
  () => import("../next/routes/inbox-page"),
  "InboxPage",
)
const SessionDetailPage = lazyRouteComponent(
  () => import("../next/routes/session/session-entry"),
  "SessionEntry",
)
const SessionsPage = lazyRouteComponent(
  () => import("../next/routes/sessions-entry"),
  "SessionsEntry",
)
const TasksPage = lazyRouteComponent(() => import("../next/routes/tasks-page"), "TasksPage")
const TaskPage = lazyRouteComponent(() => import("../next/routes/task-page"), "TaskPage")
const UsagePage = lazyRouteComponent(() => import("../next/routes/usage-page"), "UsagePage")
const SystemPage = lazyRouteComponent(() => import("../next/routes/system-page"), "SystemPage")
const TASK_SEARCH_KEYS = [
  "q",
  "group",
  "type",
  "agent",
  "assigned_agent_name",
  "session_id",
  "cayu",
  "task_id",
] as const

function RoutePending() {
  const pageLayout = useContext(ShellContext)?.pageLayout
  const placeholder = (
    <div className="space-y-4" role="status" aria-live="polite">
      <div className="h-8 w-48 animate-pulse rounded bg-muted" />
      <div className="h-32 animate-pulse rounded-lg border border-border bg-muted/40" />
      <span className="sr-only">Loading page</span>
    </div>
  )
  return pageLayout ? (
    <>
      <TopBar>Loading page</TopBar>
      <PageBody>{placeholder}</PageBody>
    </>
  ) : (
    placeholder
  )
}

export function createDashboardRouter(history?: RouterHistory) {
  const rootRoute = createRootRoute({ component: Layout })

  const dashboardRoute = createRoute({
    staticData: { dashboardLayout: "page" },
    getParentRoute: () => rootRoute,
    path: "/",
    validateSearch: (search: Record<string, unknown>): { start?: string } => ({
      start: search.start == null ? undefined : String(search.start),
    }),
    component: DashboardPage,
  })

  const sessionsRoute = createRoute({
    staticData: { dashboardLayout: "page" },
    getParentRoute: () => rootRoute,
    path: "/sessions",
    validateSearch: validateSessionIndexSearch,
    component: SessionsPage,
  })

  const usageRoute = createRoute({
    staticData: { dashboardLayout: "page" },
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
    staticData: { dashboardLayout: "page" },
    getParentRoute: () => rootRoute,
    path: "/evals",
    validateSearch: validateEvalsSearch,
    beforeLoad: ({ search }) => {
      if (search.run || search.result)
        throw redirect({
          to: "/evals/runs/$runId",
          params: {
            runId: search.run ? nativeRunSelector(search.run) : resultSelector(search.result!),
          },
          search: {
            compare: search.baseline ? resultSelector(search.baseline) : undefined,
            placement: undefined,
            filter: undefined,
            case: undefined,
          },
          replace: true,
        })
      if (search.corpus && search.suite && search.target)
        throw redirect({
          to: "/evals/suites/$suiteId",
          params: {
            suiteId: suiteSelector({
              source: "corpus",
              target_key: search.target,
              revision: search.corpus,
              suite_id: search.suite,
            }),
          },
          replace: true,
        })
    },
    component: () => (
      <CapabilityRoute requirement={DASHBOARD_ROUTE_REQUIREMENTS["/evals"]} title="Evals">
        <EvalsPage />
      </CapabilityRoute>
    ),
  })
  const evalSuiteRoute = createRoute({
    staticData: { dashboardLayout: "page" },
    getParentRoute: () => rootRoute,
    path: "/evals/suites/$suiteId",
    remountDeps: ({ params }) => params,
    component: () => (
      <CapabilityRoute requirement={DASHBOARD_ROUTE_REQUIREMENTS["/evals"]} title="Evals">
        <EvalSuitePage />
      </CapabilityRoute>
    ),
  })
  const editorComponent = () => (
    <CapabilityRoute requirement={DASHBOARD_ROUTE_REQUIREMENTS["/evals"]} title="Evals">
      <EvalSuiteEditorPage />
    </CapabilityRoute>
  )
  const evalSuiteNewRoute = createRoute({
    staticData: { dashboardLayout: "page" },
    getParentRoute: () => rootRoute,
    path: "/evals/suites/new",
    validateSearch: (s: Record<string, unknown>): { target?: string } => ({
      target: typeof s.target === "string" ? s.target : undefined,
    }),
    component: editorComponent,
  })
  const evalSuiteEditRoute = createRoute({
    staticData: { dashboardLayout: "page" },
    getParentRoute: () => rootRoute,
    path: "/evals/suites/$suiteId/edit",
    remountDeps: ({ params }) => params,
    validateSearch: (s: Record<string, unknown>): { case?: string; run?: true } => ({
      case: typeof s.case === "string" ? s.case : undefined,
      run: s.run === "true" || s.run === true ? true : undefined,
    }),
    component: editorComponent,
  })
  const evalRunRoute = createRoute({
    staticData: { dashboardLayout: "page" },
    getParentRoute: () => rootRoute,
    path: "/evals/runs/$runId",
    remountDeps: ({ params }) => params,
    validateSearch: (
      s: Record<string, unknown>,
    ): {
      compare?: string
      placement?: string
      filter?: ReturnType<typeof evalCaseFilter>
      case?: string
    } => ({
      compare: typeof s.compare === "string" ? s.compare : undefined,
      placement: typeof s.placement === "string" ? s.placement : undefined,
      filter: evalCaseFilter(s.filter),
      case: typeof s.case === "string" ? s.case : undefined,
    }),
    component: () => (
      <CapabilityRoute requirement={DASHBOARD_ROUTE_REQUIREMENTS["/evals"]} title="Evals">
        <EvalRunPage />
      </CapabilityRoute>
    ),
  })
  const evalSettingsRoute = createRoute({
    staticData: { dashboardLayout: "page" },
    getParentRoute: () => rootRoute,
    path: "/evals/settings",
    validateSearch: (search: SearchSchemaInput & { tool?: EvalSettingsTool; target?: string }) =>
      validateEvalSettingsSearch(search),
    component: () => (
      <CapabilityRoute requirement={DASHBOARD_ROUTE_REQUIREMENTS["/evals"]} title="Evals">
        <EvalSettingsPage />
      </CapabilityRoute>
    ),
  })

  const tasksRoute = createRoute({
    staticData: { dashboardLayout: "page" },
    getParentRoute: () => rootRoute,
    path: "/tasks",
    validateSearch: (
      search: Record<string, unknown>,
    ): Partial<Record<(typeof TASK_SEARCH_KEYS)[number], string>> =>
      Object.fromEntries(
        TASK_SEARCH_KEYS.filter((key) => typeof search[key] === "string" && search[key] !== "").map(
          (key) => [key, search[key]],
        ),
      ),
    component: () => (
      <CapabilityRoute requirement={DASHBOARD_ROUTE_REQUIREMENTS["/tasks"]} title="Tasks">
        <TasksPage />
      </CapabilityRoute>
    ),
  })

  const taskRoute = createRoute({
    staticData: { dashboardLayout: "page" },
    getParentRoute: () => rootRoute,
    path: "/tasks/$taskId",
    component: () => (
      <CapabilityRoute requirement={DASHBOARD_ROUTE_REQUIREMENTS["/tasks"]} title="Task">
        <TaskPage />
      </CapabilityRoute>
    ),
  })

  const pendingActionsRoute = createRoute({
    staticData: { dashboardLayout: "page" },
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

  const specialistSearch = (search: Record<string, unknown>): { q?: string } => ({
    q: typeof search.q === "string" ? search.q : undefined,
  })
  const agentsRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: "/agents",
    validateSearch: specialistSearch,
    beforeLoad: ({ search }) => {
      throw redirect({ to: "/specialists", search, replace: true })
    },
  })
  const specialistsRoute = createRoute({
    staticData: { dashboardLayout: "page" },
    getParentRoute: () => rootRoute,
    path: "/specialists",
    validateSearch: specialistSearch,
    component: SpecialistsPage,
  })
  const toolsRoute = createRoute({
    staticData: { dashboardLayout: "page" },
    getParentRoute: () => rootRoute,
    path: "/tools",
    validateSearch: specialistSearch,
    component: ToolsPage,
  })

  const environmentsRoute = createRoute({
    staticData: { dashboardLayout: "page" },
    getParentRoute: () => rootRoute,
    path: "/environments",
    validateSearch: (search: Record<string, unknown>): { q?: string } => ({
      q: typeof search.q === "string" ? search.q : undefined,
    }),
    component: EnvironmentsPage,
  })

  const environmentRoute = createRoute({
    staticData: { dashboardLayout: "page" },
    getParentRoute: () => rootRoute,
    path: "/environments/$name",
    remountDeps: ({ params }) => params,
    component: EnvironmentPage,
  })

  const artifactsRoute = createRoute({
    staticData: { dashboardLayout: "page" },
    getParentRoute: () => rootRoute,
    path: "/artifacts",
    validateSearch: validateArtifactSearch,
    component: () => (
      <CapabilityRoute requirement={DASHBOARD_ROUTE_REQUIREMENTS["/artifacts"]} title="Artifacts">
        <ArtifactsPage />
      </CapabilityRoute>
    ),
  })

  const sessionDetailRoute = createRoute({
    staticData: { dashboardLayout: "page" },
    getParentRoute: () => rootRoute,
    path: "/sessions/$sessionId",
    validateSearch: validateSessionDetailSearch,
    remountDeps: ({ params }) => ({ sessionId: params.sessionId }),
    component: SessionDetailPage,
  })

  const workflowRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: "/sessions/$sessionId/workflow",
    validateSearch: validateWorkflowSearch,
    remountDeps: ({ params }) => ({ sessionId: params.sessionId }),
    beforeLoad: ({ params, search }) => {
      throw redirect({
        to: "/sessions/$sessionId",
        params: { sessionId: params.sessionId },
        search: { ...workflowSearchPatch(search), view: "trace", span: "session", workflow: "1" },
        replace: true,
      })
    },
  })

  const knowledgeRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: "/knowledge",
    staticData: { dashboardLayout: "page" },
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
    staticData: { dashboardLayout: "page" },
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
    evalSuiteRoute,
    evalSuiteNewRoute,
    evalSuiteEditRoute,
    evalRunRoute,
    evalSettingsRoute,
    usageRoute,
    tasksRoute,
    taskRoute,
    pendingActionsRoute,
    oldPendingRoute,
    agentsRoute,
    specialistsRoute,
    toolsRoute,
    environmentsRoute,
    environmentRoute,
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
