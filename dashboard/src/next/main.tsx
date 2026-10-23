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
import { createDashboardRouter } from "./app/routes"
import { ServicesProvider } from "./app/services"
import { createMockApi } from "./mock/api"
import { installLegacyApi } from "./mock/legacy"

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
      prototype: {
        seeds: api.seeds,
        seedId,
        setSeed: (id: string) => {
          api.setSeed(id)
          setSeedId(id)
          queryClient.clear()
          void router.navigate({ to: "/" as never })
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
  return (
    <QueryClientProvider client={queryClient}>
      <ServicesProvider value={services}>
        <RouterProvider router={router} />
      </ServicesProvider>
    </QueryClientProvider>
  )
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
