/**
 * Today's pages that the rework keeps unchanged (Usage, Environments, Artifacts, System),
 * mounted as they are inside the new shell. They're bound to today's router, so they run
 * in a second router that shares the browser history: both see the same URL, and a link
 * to anything else (a session, say) is picked up by the new router.
 */
import { useQuery } from "@tanstack/react-query"
import {
  createRootRoute,
  createRoute,
  createRouter,
  Outlet,
  type RouterHistory,
  RouterProvider,
  useRouter,
} from "@tanstack/react-router"
import { CapabilityRoute, ServerContractProvider } from "@/components/dashboard/server-contract"
import { ViewerProvider } from "@/components/dashboard/viewer"
import { fetchServerContract, isSupportedServerContract } from "@/lib/api"
import { DASHBOARD_ROUTE_REQUIREMENTS } from "@/lib/dashboard-capabilities"
import { parseDashboardSearch, stringifyDashboardSearch } from "@/lib/search-params"
import { validateUsageRollupSearch } from "@/lib/usage-rollup-search"
import { ArtifactsPage } from "@/routes/artifacts"
import { EnvironmentsPage } from "@/routes/environments"
import { SystemPage } from "@/routes/system"
import { UsagePage } from "@/routes/usage"
import { ErrorState, TopBar } from "./common"

/** The server contract gate today's layout puts around every page. */
function ContractGate() {
  const contract = useQuery({ queryKey: ["server-contract"], queryFn: fetchServerContract })
  if (contract.error) return <ErrorState error={contract.error} />
  if (!contract.data) return null
  if (!isSupportedServerContract(contract.data))
    return <ErrorState error={new Error("This Cayu server's contract isn't supported.")} />
  return (
    <ServerContractProvider contract={contract.data}>
      <ViewerProvider
        value={{ name: "Prototype", runtime_version: "prototype", role: null, full_access: true }}
      >
        <Outlet />
      </ViewerProvider>
    </ServerContractProvider>
  )
}

function createLegacyRouter(history: RouterHistory) {
  const root = createRootRoute({ component: ContractGate, notFoundComponent: () => null })
  const routes = [
    createRoute({
      getParentRoute: () => root,
      path: "/usage",
      validateSearch: validateUsageRollupSearch,
      component: () => (
        <CapabilityRoute requirement={DASHBOARD_ROUTE_REQUIREMENTS["/usage"]} title="Usage">
          <UsagePage />
        </CapabilityRoute>
      ),
    }),
    createRoute({ getParentRoute: () => root, path: "/environments", component: EnvironmentsPage }),
    createRoute({
      getParentRoute: () => root,
      path: "/artifacts",
      component: () => (
        <CapabilityRoute requirement={DASHBOARD_ROUTE_REQUIREMENTS["/artifacts"]} title="Artifacts">
          <ArtifactsPage />
        </CapabilityRoute>
      ),
    }),
    createRoute({ getParentRoute: () => root, path: "/system", component: SystemPage }),
  ]
  return createRouter({
    routeTree: root.addChildren(routes),
    history,
    parseSearch: parseDashboardSearch,
    stringifySearch: stringifyDashboardSearch,
  })
}

const routers = new WeakMap<RouterHistory, ReturnType<typeof createLegacyRouter>>()

/** One of today's pages, unchanged, under the new shell's top bar. */
export function LegacyPage({ title }: { title: string }) {
  const { history } = useRouter()
  let router = routers.get(history)
  if (!router) {
    router = createLegacyRouter(history)
    routers.set(history, router)
  }
  return (
    <div className="flex min-w-0 flex-1 flex-col">
      <TopBar>
        {title}
        <span className="ml-2 rounded-full border border-border px-2 py-0.5 text-[11px] font-normal text-muted-foreground">
          Today's page, unchanged
        </span>
      </TopBar>
      <div className="min-h-0 flex-1 overflow-auto">
        <div className="p-4 sm:p-6 xl:p-8">
          <RouterProvider router={router} />
        </div>
      </div>
    </div>
  )
}
