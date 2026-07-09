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
  LayoutDashboard,
  List,
  ListTodo,
  Play,
} from "lucide-react"
import {
  fetchServerContract,
  isSupportedServerContract,
  SUPPORTED_SERVER_CONTRACT_VERSION,
} from "./lib/api"
import { dashboardAsset } from "./lib/config"
import { cn } from "./lib/utils"

const navSections = [
  {
    label: "Operate",
    items: [
      { to: "/", label: "Dashboard", icon: LayoutDashboard },
      { to: "/sessions", label: "Sessions", icon: List },
      { to: "/tasks", label: "Tasks", icon: ListTodo },
      { to: "/pending-actions", label: "Pending", icon: CircleAlert },
      { to: "/usage", label: "Usage", icon: BarChart3 },
    ],
  },
  {
    label: "Knowledge",
    items: [{ to: "/knowledge", label: "Knowledge", icon: BookOpenCheck }],
  },
  {
    label: "Runtime",
    items: [
      { to: "/agents", label: "Agents", icon: Bot },
      { to: "/environments", label: "Environments", icon: Boxes },
      { to: "/artifacts", label: "Artifacts", icon: FileArchive },
    ],
  },
  {
    label: "Create",
    items: [{ to: "/run", label: "New Run", icon: Play }],
  },
] as const

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

  return (
    <div className="flex h-screen overflow-hidden">
      <nav className="w-56 flex-shrink-0 border-r border-border bg-sidebar-background flex flex-col p-4 gap-1">
        <div className="mb-6 px-3">
          <img src={dashboardAsset("logo.svg")} alt="cayu" className="h-6" />
        </div>
        {navSections.map((section, sectionIndex) => (
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
          {(contractError || incompatibleContract) && (
            <div className="mb-6 flex items-start gap-3 rounded-md border border-destructive/30 bg-destructive/5 px-4 py-3 text-sm text-destructive">
              <AlertTriangle className="mt-0.5 h-4 w-4 flex-shrink-0" />
              <div>
                {incompatibleContract ? (
                  <>
                    Dashboard expects CAYU server contract v{SUPPORTED_SERVER_CONTRACT_VERSION}, but
                    the server reports v{contract.data.contract_version}.
                  </>
                ) : (
                  <>Could not load the CAYU server contract: {contractError}</>
                )}
              </div>
            </div>
          )}
          <Outlet />
        </div>
      </main>
    </div>
  )
}
