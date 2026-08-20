import { useQuery } from "@tanstack/react-query"
import { Link, Outlet, useRouterState } from "@tanstack/react-router"
import {
  AlertTriangle,
  BarChart3,
  BookOpenCheck,
  Bot,
  Boxes,
  CircleAlert,
  FileArchive,
  FlaskConical,
  LayoutDashboard,
  List,
  ListTodo,
  Play,
  Settings2,
} from "lucide-react"
import { ServerContractProvider } from "./components/dashboard/server-contract"
import { Button } from "./components/ui/button"
import {
  fetchServerContract,
  isSupportedServerContract,
  SUPPORTED_SERVER_CONTRACT_VERSION,
} from "./lib/api"
import { dashboardAsset } from "./lib/config"
import {
  DASHBOARD_ROUTE_REQUIREMENTS,
  type DashboardCapabilityRequirement,
  dashboardCapabilityEnabled,
} from "./lib/dashboard-capabilities"
import type { ControlPlaneCapabilities } from "./lib/generated/server-api"
import { cn } from "./lib/utils"

const navSections = [
  {
    label: "Operate",
    items: [
      {
        to: "/",
        label: "Dashboard",
        icon: LayoutDashboard,
        capability: DASHBOARD_ROUTE_REQUIREMENTS["/"],
      },
      { to: "/sessions", label: "Sessions", icon: List },
      {
        to: "/evals",
        label: "Evals",
        icon: FlaskConical,
      },
      {
        to: "/tasks",
        label: "Tasks",
        icon: ListTodo,
        capability: DASHBOARD_ROUTE_REQUIREMENTS["/tasks"],
      },
      { to: "/pending-actions", label: "Pending", icon: CircleAlert },
      {
        to: "/usage",
        label: "Usage",
        icon: BarChart3,
        capability: DASHBOARD_ROUTE_REQUIREMENTS["/usage"],
      },
    ],
  },
  {
    label: "Knowledge",
    items: [
      {
        to: "/knowledge",
        label: "Knowledge",
        icon: BookOpenCheck,
        capability: DASHBOARD_ROUTE_REQUIREMENTS["/knowledge"],
      },
    ],
  },
  {
    label: "Runtime",
    items: [
      { to: "/agents", label: "Agents", icon: Bot },
      { to: "/environments", label: "Environments", icon: Boxes },
      {
        to: "/artifacts",
        label: "Artifacts",
        icon: FileArchive,
        capability: DASHBOARD_ROUTE_REQUIREMENTS["/artifacts"],
      },
      { to: "/system", label: "System", icon: Settings2 },
    ],
  },
  {
    label: "Create",
    items: [
      {
        to: "/run",
        label: "New Run",
        icon: Play,
        capability: DASHBOARD_ROUTE_REQUIREMENTS["/run"],
      },
    ],
  },
] as const

function navItemAvailable(
  capabilities: ControlPlaneCapabilities,
  capability?: DashboardCapabilityRequirement,
) {
  return capability === undefined || dashboardCapabilityEnabled(capabilities, capability)
}

export function Layout() {
  const router = useRouterState()
  const currentPath = router.location.pathname
  const contract = useQuery({
    queryKey: ["server-contract"],
    queryFn: fetchServerContract,
    refetchInterval: false,
    staleTime: Number.POSITIVE_INFINITY,
  })
  const contractError = contract.error instanceof Error ? contract.error.message : null
  const incompatibleContract =
    contract.data !== undefined && !isSupportedServerContract(contract.data)
  const visibleNavSections = navSections
    .map((section) => ({
      ...section,
      items: section.items.filter((item) => {
        const capability = "capability" in item ? item.capability : undefined
        if (capability === undefined) return true
        if (contract.data === undefined || incompatibleContract) return false
        return navItemAvailable(contract.data.capabilities, capability)
      }),
    }))
    .filter((section) => section.items.length > 0)

  return (
    <div className="flex h-screen overflow-hidden">
      <nav className="w-56 flex-shrink-0 border-r border-border bg-sidebar-background flex flex-col p-4 gap-1">
        <div className="mb-6 px-3">
          <img src={dashboardAsset("logo.svg")} alt="cayu" className="h-6" />
        </div>
        {visibleNavSections.map((section, sectionIndex) => (
          <div
            key={section.label}
            className={cn("space-y-1", sectionIndex > 0 && "mt-3 border-t border-border pt-3")}
          >
            {section.items.map(({ to, label, icon: Icon }) => {
              const active = to === "/" ? currentPath === "/" : currentPath.startsWith(to)
              return (
                <Link
                  key={to}
                  to={to}
                  className={cn(
                    "flex items-center gap-3 px-3 py-2 rounded-md text-sm no-underline transition-colors",
                    active
                      ? "bg-sidebar-accent text-sidebar-accent-foreground font-medium"
                      : "text-muted-foreground hover:bg-sidebar-accent/50 hover:text-sidebar-accent-foreground",
                  )}
                >
                  <Icon size={18} />
                  {label}
                </Link>
              )
            })}
          </div>
        ))}
        <div className="mt-auto rounded-md border border-sidebar-border bg-background/70 px-3 py-2 text-xs text-muted-foreground">
          {contract.isLoading ? (
            <span>Checking API contract...</span>
          ) : contractError ? (
            <span className="text-destructive">API contract unavailable</span>
          ) : contract.data ? (
            <span>
              API {contract.data.api_prefix} · v{contract.data.contract_version}
            </span>
          ) : null}
        </div>
      </nav>
      <main className="min-w-0 flex-1 overflow-auto">
        <div className="p-4 sm:p-6 xl:p-8">
          {contract.isLoading ? (
            <div className="flex items-center gap-3 rounded-md border border-border bg-muted/40 px-4 py-3 text-sm text-muted-foreground">
              Checking API contract before loading control-plane data...
            </div>
          ) : contractError || incompatibleContract ? (
            <div
              className="mb-6 flex items-start gap-3 rounded-md border border-destructive/30 bg-destructive/5 px-4 py-3 text-sm text-destructive"
              data-testid="dashboard-contract-gate"
              role="alert"
            >
              <AlertTriangle className="mt-0.5 h-4 w-4 flex-shrink-0" />
              <div className="space-y-2">
                {incompatibleContract
                  ? `Dashboard expects CAYU server contract v${SUPPORTED_SERVER_CONTRACT_VERSION}, but the server reports v${contract.data.contract_version}.`
                  : `Could not load the CAYU server contract: ${contractError}`}
                <div>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    disabled={contract.isFetching}
                    onClick={() => void contract.refetch()}
                  >
                    {contract.isFetching ? "Checking..." : "Retry contract check"}
                  </Button>
                </div>
              </div>
            </div>
          ) : contract.data ? (
            <ServerContractProvider contract={contract.data}>
              <Outlet />
            </ServerContractProvider>
          ) : (
            <div role="alert">The CAYU server contract returned no data.</div>
          )}
        </div>
      </main>
    </div>
  )
}
