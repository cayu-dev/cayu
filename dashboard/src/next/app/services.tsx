import { createContext, type ReactNode, useContext } from "react"
import type { Api } from "../api/types"

/** Present only in the prototype: switch between seeded apps and reset the fake data. */
export type PrototypeControls = {
  seeds: { id: string; label: string; description: string }[]
  seedId: string
  setSeed: (id: string) => void
  reset: () => void
  appTitles: boolean
  setAppTitles: (on: boolean) => void
}

type Services = {
  api: Api
  /** The app's clock. The prototype pins it to its seed; the real dashboard uses Date.now. */
  now: () => number
  prototype?: PrototypeControls
  overviewQueriesEnabled?: boolean
  usageSeriesEnabled?: boolean
  sessionOverviewSeriesEnabled?: boolean
  artifactsEnabled?: boolean
  taskIndexEnabled?: boolean
  taskCreationEnabled?: boolean
  taskMutationsEnabled?: boolean
  executionEnabled?: boolean
  inboxEnabled?: boolean
  decisionsEnabled?: boolean
  knowledgeReviewEnabled?: boolean
}

const ServicesContext = createContext<Services | null>(null)

export function ServicesProvider({ value, children }: { value: Services; children: ReactNode }) {
  return <ServicesContext.Provider value={value}>{children}</ServicesContext.Provider>
}

export function useServices() {
  const services = useContext(ServicesContext)
  if (!services) throw new Error("useServices must be used inside <ServicesProvider>.")
  return services
}

export const useApi = () => useServices().api
export const useNow = () => useServices().now
