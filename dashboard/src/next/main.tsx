/**
 * The reworked Developer dashboard as a prototype: its real pages and components,
 * answered by fake data (mock/api.ts) instead of /api. Hash routing, so the build
 * works from any folder and inside a shared preview.
 */
import "./legacy-config"
import "./next.css"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { createHashHistory, RouterProvider } from "@tanstack/react-router"
import { StrictMode, useMemo, useState } from "react"
import { createRoot } from "react-dom/client"
import { ServerContractProvider } from "@/components/dashboard/server-contract"
import { ViewerProvider } from "@/components/dashboard/viewer"
import type { AppInfoResponse } from "@/lib/generated/server-api"
import { DashboardPartsHost } from "./app/parts-host"
import { createDashboardRouter } from "./app/routes"
import { ServicesProvider } from "./app/services"
import { createMockApi } from "./mock/api"
import { installLegacyApi } from "./mock/legacy"
import { CONTRACT } from "./mock/legacy/contract"

const api = createMockApi()
// Today's unchanged pages call /api through fetch; answer them from the same seed.
installLegacyApi(api)
const queryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 10_000, retry: 1, refetchOnWindowFocus: false },
    mutations: { retry: 0 },
  },
})
const router = createDashboardRouter(createHashHistory())

function App() {
  const [seedId, setSeedId] = useState(api.seedId())
  const [appTitles, setAppTitles] = useState(true)
  const services = useMemo(
    () => ({
      api,
      now: api.now,
      environmentDetailsEnabled: true,
      artifactSearchEnabled: true,
      usageCostDetailsEnabled: true,
      usageSeriesEnabled: true,
      sessionOverviewSeriesEnabled: true,
      sessionOverviewCountFiltersEnabled: true,
      overviewFirstMessageEnabled: true,
      prototype: {
        seeds: api.seeds,
        seedId,
        setSeed: (id: string) => {
          api.setSeed(id)
          setSeedId(id)
          queryClient.clear()
          void router.navigate({ to: "/" })
        },
        reset: () => {
          api.reset()
          void queryClient.resetQueries()
        },
        appTitles,
        setAppTitles: (on: boolean) => {
          api.setAppTitles(on)
          setAppTitles(on)
          void queryClient.invalidateQueries()
        },
      },
    }),
    [seedId, appTitles],
  )
  // The production shell mounts pages only after its contract and identity gates pass;
  // stand in for both with the fake server's contract and an open local deployment's viewer.
  // biome-ignore lint/correctness/useExhaustiveDependencies: the viewer follows the seed.
  const viewer = useMemo<AppInfoResponse>(() => {
    const { app } = api.records()
    return {
      name: app.name,
      runtime_version: app.runtime_version,
      full_access: true,
      role: null,
      human_review_required: false,
      human_review_purpose: null,
    }
  }, [seedId])
  return (
    <QueryClientProvider client={queryClient}>
      <ServicesProvider value={services}>
        <ServerContractProvider contract={CONTRACT}>
          <ViewerProvider value={viewer}>
            <DashboardPartsHost>
              <RouterProvider router={router} />
            </DashboardPartsHost>
          </ViewerProvider>
        </ServerContractProvider>
      </ServicesProvider>
    </QueryClientProvider>
  )
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
